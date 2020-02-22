"""
Sudoku solution tactics.  These include the
constraint propogation tactics and (in phase
two) the search-based solver.

Author: Henzi Kou
"""

from sdk_board import Board
import sdk_tile

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def naked_single(board: Board) -> bool:
    """As described in http://www.sadmansoftware.com/sudoku/nakedsingle.php
    Returns True iff some change has been made
    """
    logging.info("Applying naked single tactic")
    changed = False
    for group in board.groups:
        changed = group.naked_single_constrain() or changed
    return changed


def hidden_single(board: Board) -> bool:
    """As described in http://www.sadmansoftware.com/sudoku/hiddensingle.php
    Returns True iff some change has been made
    """
    logging.info("Applying hidden single tactic")
    changed = False
    for group in board.groups:
        changed = group.hidden_single_constrain() or changed
    return changed


def propagate(board: Board):
    """Propagate constraints until we either solve the puzzle,
    show the puzzle as given is unsolvable, or can make no more
    progress by constraint propagation.
    """
    logging.info("Propagating constraints")
    changed = True
    while changed:
        logging.info("Invoking naked single")
        changed = naked_single(board)
        if board.is_solved() or not board.is_consistent():
            return
        changed = hidden_single(board) or changed
        if board.is_solved() or not board.is_consistent():
            return
    return


def solve(board: Board) -> bool:
    """Main solver.  Initially this just invokes constraint
    propagation.  In phase 2 of the project, you will add
    recursive back-tracking search (guess-and-check with recursion).
    """
    log.debug("Called solve on board:\n{}".format(board))
    propagate(board)

    if board.is_solved():
        return True
    if not board.is_consistent():
        return False

    min_cands = 10
    best_tile = None
    save = board.as_list()

    for row in board.tiles:
        for tile in row:
            if tile.value == sdk_tile.UNKNOWN and len(tile.candidates) < min_cands:
                best_tile = tile
                min_cands = len(tile.candidates)

    for cands in best_tile.candidates:
        best_tile.set_value(cands)
        if solve(board):
            return True
        else:
            board.set_tiles(save)

    return False
