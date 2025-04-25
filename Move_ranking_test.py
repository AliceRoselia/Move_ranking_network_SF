# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 23:46:21 2025

@author: User
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import pandas as pd
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


# Main execution
# Inference example
def rank_moves(board, legal_moves, model):
    
    model.eval()
    with torch.no_grad():
        scores = model(board, legal_moves)
        move_scores = [(m, s.item()) for m, s in zip(legal_moves, scores)]
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return move_scores

model = torch.load("my_best_network.pt", weights_only=False)
    
    
puzzles = pd.read_csv("lichess_puzzle.csv")
answer_correct = 0
min_max_differences = []
chosen_max_differences = []
too_tight = 0
top_3 = 0

puzzles = puzzles[puzzles["Rating"]<=1200]
for i,j in puzzles[["FEN","Moves"]][:10000].iterrows():
    board = chess.Board(j.FEN)
    flip_moves = False
    if board.turn == chess.BLACK:
        board = board.mirror()
        flip_moves = True
    legal_moves = list(board.legal_moves)
    ranked_moves = rank_moves(board, legal_moves, model)
    if flip_moves:
        for x,y in ranked_moves:
            flip_move(x)
        board = board.mirror()
    
    move_rankings = {m.uci(): s for m, s in ranked_moves}
    max_moves = ranked_moves[0][1]
    min_moves = ranked_moves[-1][1]
    
    chosen_move = ranked_moves[0][0]
    value = move_rankings[j.Moves.split()[0]]
    #print(board)
    #print(value)
    #print([(i.uci(), j) for i,j in ranked_moves[:3]])
    #print("correct move:",j.Moves.split()[0])
    top_3_current = [i.uci() for i,j in ranked_moves[:3]]
    top_3 += (j.Moves.split()[0] in top_3_current)
    min_max_differences.append(max_moves-min_moves)
    chosen_max_differences.append(max_moves-value)
    
import numpy as np
min_max_differences = np.array(min_max_differences)
chosen_max_differences = np.array(chosen_max_differences)
answer_correct = (chosen_max_differences == 0.0).sum()
print(answer_correct)
print(top_3)
print(min_max_differences.mean())
print(chosen_max_differences.mean())