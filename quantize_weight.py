

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import chess
import numpy as np
from math import sqrt

# Model definition
def flip_move(move):
    from_square = (move.from_square & 7) | (56 - (move.from_square & 56))
    to_square = (move.to_square & 7) | (56 - (move.to_square & 56))
    return chess.Move(from_square, to_square, promotion=move.promotion)

class MoveRankingModel(nn.Module):
    def __init__(self, vector_size=16, vector_size_2=8):
        super().__init__()
        self.piece_square_vectors = nn.Parameter(torch.randn(12, 64, vector_size) / sqrt(vector_size))
        self.move_vectors = nn.Parameter(torch.randn(6, 64, vector_size, vector_size_2) / sqrt(vector_size))
        self.piece_square_bias = nn.Parameter(torch.randn(vector_size) / sqrt(vector_size))
        self.bias2 = nn.Parameter(torch.randn(6, 64, vector_size_2))
        self.output_layer = nn.Parameter(torch.randn(6, 64, vector_size_2) / vector_size_2)
        self.output_bias = nn.Parameter(torch.zeros(6, 64))

    def compute_board_representation(self, board):
        b = self.piece_square_bias.clone()
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_idx = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
                b += self.piece_square_vectors[piece_idx, square]
        return b

    def forward(self, board, legal_moves):
        b = self.compute_board_representation(board)
        move_scores = []
        for move in legal_moves:
            piece = board.piece_at(move.from_square)
            piece_idx = piece.piece_type - 1
            to_square = move.to_square
            move_vec = self.move_vectors[piece_idx, to_square]
            hidden = F.relu(self.bias2[piece_idx, to_square] + torch.matmul(b, move_vec))
            score = (torch.dot(hidden, self.output_layer[piece_idx, to_square]) + 
                     self.output_bias[piece_idx, to_square])
            move_scores.append(score)
        return torch.stack(move_scores)

# Quantization function
def quantize_to_int16(tensor, scale_factor=None):
    """Quantize a tensor to int16 with symmetric linear quantization."""
    if scale_factor is None:
        max_abs = torch.max(torch.abs(tensor))
        scale_factor = max_abs / 32767 if max_abs > 0 else 1.0
    quantized = torch.round(tensor / scale_factor).clamp(-32768, 32767).to(torch.int16)
    return quantized, scale_factor

def tostring(value):
    if len(value.shape) == 1:
        return "{" + ",".join(str(i) for i in value) + "}\n"
    else:
        return "{" + ",".join(tostring(i) for i in value) + "}\n"
    


def to_cpp_array(vartype,varname,values,f):
    f.write(vartype)
    f.write(" ")
    f.write(varname)
    f.write("".join(f"[{i}]" for i in values.shape))
    f.write("=")
    f.write(tostring(values).strip())
    f.write(";\n")

# Export quantized weights
def export_quantized_weights(model, filename="weights_quantized.h"):
    with open(filename, "w") as f:
        scale = 1.5322 #Check max absolute value.
        quantized_piece_square_vector = (32767 * model.piece_square_vectors.detach().numpy()/scale).astype(np.int16)
        quantized_piece_square_bias = (32767 * model.piece_square_bias.detach().numpy()/scale).astype(np.int16)

        to_cpp_array("constexpr int16_t", "piece_square_vectors", quantized_piece_square_vector, f)
        to_cpp_array("constexpr int16_t", "piece_square_bias", quantized_piece_square_bias, f)
        
        
        # At this point, scale = 13008.456072094963, meaning 13008 means 1.
        quantized_move_vectors = ((512 * model.move_vectors.detach().numpy())).astype(np.int16)
        quantized_bias2 = (512*32767 * model.bias2.detach().numpy()/scale).astype(np.int32)
        
        #At this point, scale = 213130544.28520387

        to_cpp_array("constexpr int32_t","bias2",quantized_bias2,f)
        to_cpp_array("constexpr int16_t", "move_vectors", quantized_move_vectors, f)
        
        #1.0380
        scale2 = 0.6366
        quantized_output_layer = ((32767 * model.output_layer.detach().numpy()/scale2).astype(np.int16))
        #At this point, we will shift by 16 bits.
        #final_scale = 3252.1140180237408
        #Scaled up to 6416306.60095038
        quantized_output_bias = ((32767/scale*512/(2**16)*32767/scale2) * model.output_bias.detach().numpy()).astype(np.int32)
        to_cpp_array("constexpr int16_t","output_layer",quantized_output_layer,f)
        to_cpp_array("constexpr int32_t", "output_bias", quantized_output_bias, f)
        

# Validation
def evaluate_model(model, puzzles, max_positions=10000):
    model.eval()
    answer_correct = 0
    top_3 = 0
    min_max_differences = []
    chosen_max_differences = []
    with torch.no_grad():
        for i, j in puzzles[["FEN", "Moves"]][:max_positions].iterrows():
            board = chess.Board(j.FEN)
            if board.turn == chess.BLACK:
                board.push(chess.Move.from_uci(j.Moves.split()[0]))
                solution = j.Moves.split()[1]
            else:
                solution = j.Moves.split()[0]
            legal_moves = list(board.legal_moves)
            ranked_moves = rank_moves(board, legal_moves, model)
            
            move_rankings = {m.uci(): s for m, s in ranked_moves}
            max_moves = ranked_moves[0][1]
            min_moves = ranked_moves[-1][1]
            chosen_move = ranked_moves[0][0]
            value = move_rankings.get(solution, min_moves)
            
            top_3_current = [m.uci() for m, s in ranked_moves[:3]]
            top_3 += (solution in top_3_current)
            min_max_differences.append(max_moves - min_moves)
            chosen_max_differences.append(max_moves - value)
            
            if chosen_move.uci() == solution:
                answer_correct += 1
                
    min_max_differences = np.array(min_max_differences)
    chosen_max_differences = np.array(chosen_max_differences)
    top_1_accuracy = answer_correct / max_positions
    top_3_accuracy = top_3 / max_positions
    print(f"Top-1 Accuracy: {top_1_accuracy:.4f}")
    print(f"Top-3 Accuracy: {top_3_accuracy:.4f}")
    print(f"Mean Min-Max Difference: {min_max_differences.mean():.4f}")
    print(f"Mean Chosen-Max Difference: {chosen_max_differences.mean():.4f}")
    return top_1_accuracy, top_3_accuracy

# Inference
def rank_moves(board, legal_moves, model):
    model.eval()
    with torch.no_grad():
        scores = model(board, legal_moves)
        move_scores = [(m, s.item()) for m, s in zip(legal_moves, scores)]
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return move_scores

# Main
if __name__ == "__main__":
    model = torch.load("moveranking_model_7_epoch0.pt", weights_only=False)
    #model.eval()

    #puzzles = pd.read_csv("lichess_puzzle.csv")
    #print("Evaluating original model...")
    #top1, top3 = evaluate_model(model, puzzles)

    print("Quantizing weights...")
    export_quantized_weights(model, "weights_quantized.h")