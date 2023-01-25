#!/usr/bin/python3

# from PyQt5.QtCore import QLineF, QPointF



import math
import time

# Used to compute the bandwidth for banded version
MAXINDELS = 3

# Used to implement Needleman-Wunsch scoring
MATCH = -3
INDEL = 5
SUB = 1


class GeneSequencing:

    def __init__(self):
        pass

# This is the method called by the GUI.  _sequences_ is a list of the ten sequences, _table_ is a
# handle to the GUI so it can be updated as you find results, _banded_ is a boolean that tells
# you whether you should compute a banded alignment or full alignment, and _align_length_ tells you 
# how many base pairs to use in computing the alignment
    def align(self, sequences, table, banded, align_length):
        self.banded = banded
        self.MaxCharactersToAlign = align_length
        results = []

        for i in range(len(sequences)):
            jresults = []
            for j in range(len(sequences)):
                if j < i:
                    s = {}
                else:
                    if banded:
                        score, alignment1, alignment2 = self.do_banded(sequences[i], sequences[j])
                    else:
                        score, alignment1, alignment2 = self.do_unrestricted(sequences[i], sequences[j])

                    s = {'align_cost': score, 'seqi_first100': alignment1, 'seqj_first100': alignment2}
                    table.item(i, j).setText('{}'.format(int(score) if score != math.inf else score))
                    table.repaint()    
                jresults.append(s)
            results.append(jresults)
        return results

    def do_banded(self, first_seq, second_seq):  # O(kn) time and space
        results = []  # 2-d array of resulting values, O(kn) space
        prev_dict = {}  # dictionary of back-traces, also O(kn) space
        first_length = min(len(first_seq), self.MaxCharactersToAlign)
        second_length = min(len(second_seq), self.MaxCharactersToAlign)

        if abs(first_length-second_length) > MAXINDELS:  # we already know if banding will not work
            return math.inf, "No Alignment Possible", "No Alignment Possible"

        for i in range(first_length + 1):  # loop through the first string, O(n) time
            jline = []
            nones = MAXINDELS - i
            for j in range(0, 2*MAXINDELS+1):  # O(2k + 1) time, which reduces to O(k)
                k = j + i - MAXINDELS
                if k > second_length:  # we are at the end of the table...
                    jline.append(None)
                    continue
                if nones > 0:  # insert Nones in the top left corner
                    jline.append(None)
                    nones -= 1
                    continue
                elif j == MAXINDELS and i == 0:  # starting position
                    jline.append(0)
                    prev_dict[(0, 0)] = None
                else:  # everything in the middle
                    up, diag, left = math.inf, math.inf, math.inf
                    if i > 0 and j != 2*MAXINDELS:   # check up
                        up = results[i-1][j+1] + INDEL
                    if i > 0 and results[i-1][j] is not None:  # check diagonal
                        diag = results[i-1][j]
                        if first_seq[i-1] == second_seq[k-1]:  # if MATCH
                            diag += MATCH
                        else:
                            diag += SUB
                    if j > 0 and jline[-1] is not None:  # check left
                        left = jline[-1] + INDEL

                    # choose minimum of up, diag, and left
                    jline.append(min(up, diag, left))
                    # add this to our prev_dict
                    if jline[j] == left:
                        prev_dict[(i, k)] = (i, k-1)
                    elif jline[j] == up:
                        prev_dict[(i, k)] = (i-1, k)
                    else:
                        prev_dict[(i, k)] = (i-1, k-1)
            results.append(jline)  # append the row to the table
        score = results[first_length][MAXINDELS + (second_length-first_length)]  # the final score

        # get alignments, O(n) time
        alignment1, alignment2 = self.get_alignments(first_seq, second_seq, first_length, second_length, prev_dict)

        # return first 100 characters of each alignment
        return score, alignment1[0:99], alignment2[0:99]

    def do_unrestricted(self, first_seq, second_seq):  # O(nm) time and space
        results = []  # 2-d array of resulting values, O(nm) space
        prev_dict = {}  # dictionary of back-traces, also O(nm) space
        first_length = min(len(first_seq), self.MaxCharactersToAlign)
        second_length = min(len(second_seq), self.MaxCharactersToAlign)
        for i in range(first_length+1):  # O(n)
            jline = []
            for j in range(second_length+1):  # O(m)
                if j == 0 and i == 0:  #
                    jline.append(0)
                    prev_dict[(0, 0)] = None
                elif i == 0:  # top row
                    # add INDEL
                    jline.append(jline[j-1] + INDEL)
                    prev_dict[(i, j)] = (i, j-1)
                elif j == 0:  # first column
                    # add INDEL
                    jline.append(results[i-1][j] + INDEL)
                    prev_dict[(i, j)] = (i-1, j)
                else:
                    # check up
                    up = results[i-1][j] + INDEL
                    # check diagonal
                    diag = results[i-1][j-1]
                    if first_seq[i-1] == second_seq[j-1]:
                        diag += MATCH
                    else:
                        diag += SUB
                    # check left
                    left = jline[j-1] + INDEL

                    # choose minimum of up, diag, or left
                    jline.append(min(up, diag, left))
                    # add this to our prev_dict
                    if jline[j] == left:
                        prev_dict[(i, j)] = (i, j-1)
                    elif jline[j] == up:
                        prev_dict[(i, j)] = (i-1, j)
                    else:
                        prev_dict[(i, j)] = (i-1, j-1)
            results.append(jline)  # append the row to the table
        score = results[first_length][second_length]  # the final score

        # get alignments, O(n) time
        alignment1, alignment2 = self.get_alignments(first_seq, second_seq, first_length, second_length, prev_dict)

        # return first 100 characters of each alignment
        return score, alignment1[0:99], alignment2[0:99]

    def get_alignments(self, first_seq, second_seq, first_length, second_length, prev_dict):
        # O(n) time where n is the longer of the lengths (for banded, this won't be
        #   more than MAXINDELS more than the shorter length)
        alignment1 = ""
        alignment2 = ""
        current_index = (first_length, second_length)
        while True:  # O(n)
            prev_index = prev_dict[current_index]
            if prev_index is None:
                break
            elif prev_index[0] != current_index[0] and prev_index[1] != current_index[1]:  # add letters to both
                alignment1 += first_seq[prev_index[0]]
                alignment2 += second_seq[prev_index[1]]
            elif prev_index[0] != current_index[0]:  # up
                alignment1 += first_seq[prev_index[0]]
                alignment2 += '-'
            else:  # left
                alignment1 += '-'
                alignment2 += second_seq[prev_index[1]]
            current_index = prev_index

        # the alignments were constructed backwards, flip them in O(n) time
        final_alignment1 = alignment1[::-1]
        final_alignment2 = alignment2[::-1]

        return final_alignment1, final_alignment2
