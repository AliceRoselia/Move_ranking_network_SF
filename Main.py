import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import chess
from math import sqrt

# Model definition
def flip_move(move):
    
    move.from_square = (move.from_square&7) | (56-(move.from_square&56))
    move.to_square = (move.to_square&7) | (56-(move.to_square&56))

class MoveRankingModel(nn.Module):
    def __init__(self, vector_size=16,vector_size_2 = 8):
        
        super(MoveRankingModel, self).__init__()
        # 12 piece types (6 white, 6 black), 64 squares, vector size 16
        self.piece_square_vectors = nn.Parameter(torch.randn(12, 64, vector_size)/sqrt(vector_size))
        # 6 piece types, 64 to-squares, vector size 16
        self.move_vectors = nn.Parameter(torch.randn(6, 64, vector_size, vector_size_2)/sqrt(vector_size))
        self.piece_square_bias = nn.Parameter(torch.randn(vector_size)/sqrt(vector_size))
        
        self.bias2 = nn.Parameter(torch.randn(6,64,vector_size_2))
        self.output_layer = nn.Parameter(torch.randn(6, 64, vector_size_2)/vector_size_2)
        self.output_bias = nn.Parameter(torch.zeros(6,64))

    def compute_board_representation(self, board):
        # board: chess.Board object
        b = torch.clone(self.piece_square_bias)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Map piece to index: 0-5 (white P,N,B,R,Q,K), 6-11 (black P,N,B,R,Q,K)
                piece_idx = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
                b += self.piece_square_vectors[piece_idx, square]
        return b

    def forward(self, board, legal_moves):
        # board: chess.Board object
        # legal_moves: list of chess.Move objects
        b = self.compute_board_representation(board)
        move_scores = []
        for move in legal_moves:
            # Get piece type and to-square
            piece = board.piece_at(move.from_square)
            piece_idx = piece.piece_type - 1
            to_square = move.to_square
            move_vec = self.move_vectors[piece_idx, to_square]
            hidden = F.relu(self.bias2[piece_idx, to_square] + torch.matmul(b, move_vec))
            score = (torch.dot(hidden,self.output_layer[piece_idx,to_square]) + 
            self.output_bias[piece_idx,to_square])
            
            move_scores.append(score)
        return torch.stack(move_scores)

# Dataset preparation
def prepare_dataset(file_path):
    boards = []
    legal_moves_list = []
    best_move_indices = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            
            info = line.strip().split()
            fen, uci_move = " ".join(info[:-1]), info[-1]
            board = chess.Board(fen)
            best_move = chess.Move.from_uci(uci_move)
            if (board.turn == chess.BLACK):
                board = board.mirror()
                flip_move(best_move)
            legal_moves = list(board.legal_moves)
            if best_move not in legal_moves:
                continue  # Skip invalid moves
            boards.append(board)
            legal_moves_list.append(legal_moves)
            best_move_indices.append(legal_moves.index(best_move))
    return boards, legal_moves_list, best_move_indices

# Training loop
def train_model(data_file, epochs=2):
    model = MoveRankingModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    boards, legal_moves_list, best_move_indices = prepare_dataset(data_file)

    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(boards)):
            if (i%1000 == 0):
                print(i)
            
            optimizer.zero_grad()
            scores = model(boards[i], legal_moves_list[i])
            target = torch.tensor([best_move_indices[i]], dtype=torch.long)
            loss = criterion(scores.unsqueeze(0), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch}, Average Loss: {total_loss / len(boards):.4f}")
        torch.save(model,f"moveranking_model_6_epoch{epoch}.pt")
    return model

# Inference example
def rank_moves(board, legal_moves, model):
    
    model.eval()
    with torch.no_grad():
        scores = model(board, legal_moves)
        move_scores = [(m, s.item()) for m, s in zip(legal_moves, scores)]
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return move_scores

# Main execution
if __name__ == "__main__":
    data_file = "training_data.txt"
    model = train_model(data_file)