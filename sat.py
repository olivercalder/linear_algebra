'''
Knowledge Base Satisfiability Checker
By Oliver Calder
November 2021

See the docstring of the dpll() function for a discussion of the algorithm which
is used to compute the satisfiability of knowledge bases.
'''

import sys
import time
import argparse


def read_kb_from_fd(fd):
    '''
    Reads a CNF knowledge base line-by-line from the file descriptor given by fd.
    fd should be an open file descriptor with read permissions.

    Each line in the fd corresponds to a CNF clause, and should be represented
    as a space-separated list of literals given by integers, where a negative
    integer denotes the corresponding literal is false in the given clause.

    Returns a list of clauses, where each clause is a tuple of integers.
    '''
    kb = []
    for line in fd:
        line = line.strip()
        if line[0] != '#':   # Allow comments via lines that begin with #
            line = line.replace('\t', ' ')
            while '  ' in line:
                line = line.replace('  ', ' ')
            kb.append(tuple([int(x) for x in line.strip().split(' ')]))
    return kb


def read_kb_from_file(filename):
    '''
    Reads a CNF knowledge base line-by-line from the file given by filename.
    Opens the given file with read permissions and then reads line-by-line.

    Each line in the fd corresponds to a CNF clause, and should be represented
    as a space-separated list of literals given by integers, where a negative
    integer denotes the corresponding literal is false in the given clause.

    Returns a list of clauses, where each clause is a tuple of integers.
    '''
    with open(filename) as infile:
        kb = read_kb_from_fd(infile)
    return kb


def dump_kb_to_fd(kb, fd):
    '''
    Writes the given CNF knowledge base line-by-line to the file descriptor
    given by fd. fd should be an open file descriptor with write permissions.

    Each line written to the fd corresponds to a CNF clause, and is be represented
    as a space-separated list of literals given by integers, where a negative
    integer denotes the corresponding literal is false in the given clause.
    '''
    for clause in kb:
        print(' '.join([str(var) for var in clause]), file=fd)


def dump_kb_to_file(kb, filename):
    '''
    Writes the given CNF knowledge base line-by-line to the file given by the
    given filename. Opens the given file with write permissions and then writes
    one clause at a time.

    Each line written to the fd corresponds to a CNF clause, and is be represented
    as a space-separated list of literals given by integers, where a negative
    integer denotes the corresponding literal is false in the given clause.
    '''
    with open(filename, 'w') as outfile:
        dump_kb_to_fd(kb, outfile)


def get_max_var_from_kb(kb):
    '''
    Computes the integer literal with the greatest absolute value appearing in
    the knowledge base. This is useful when converting a knowledge base to 3-SAT
    form, when new literals must be used to split clauses with more than three
    literals into multiple clauses of at most three literals.

    Returns the absolute value of the literal with greatest absolute value from
    the knowledge base.
    '''
    max_var = 0
    for clause in kb:
        if clause:
            for var in clause:
                max_var = max(max_var, abs(var))
    return max_var


def sorted_kb(kb):
    '''
    Sorts the given knowledge base according to the absolute value of the first
    literal in the clause. If the clause is empty, puts it first.

    Returns the sorted list of clause tuples.
    '''
    return sorted(kb, key=lambda clause: abs(clause[0]) if clause else -9999999999999)


def sorted_literals(literals):
    '''
    Sorts the given set of literals according to the absolute value of the literals.
    '''
    return sorted(literals, key=lambda var: abs(var))


def convert_kb_to_3sat(kb):
    '''
    Converts the given knowledge base to 3-SAT form by replacing any clause which
    contains more than three literals with several clauses containing at most
    three literals, such that the conjunction of the new clauses is logically
    equivalent to the original clause. To do so, new literals are introduced
    which do not occur elsewhere in the knowledge base.

    For example:
    (1, 2, 3) becomes (1, 2, 3)
    (1, 2, 3, 4) becomes (1, 2, 5) and (-5, 3, 4)
    (1, 2, 3, 4, 5) becomes (1, 2, 6) and (-6, 3, 7) and (-7, 4, 5)
    (1, 2, 3, 4, 5, 6) becomes (1, 2, 7) and (-7, 3, 8) and (-8, 4, 9) and (-9, 5, 6)
    etc.

    Returns the new knowledge base as a set of clauses in 3-SAT CNF.
    '''
    max_var = get_max_var_from_kb(kb)
    kb_3sat = set()
    for clause in kb:
        if len(clause) > 3:
            kb_3sat.add((clause[0], clause[1], max_var + 1))
            clause = clause[2:]
            max_var += 1
            while len(clause) > 2:
                kb_3sat.add((-max_var, clause[0], max_var + 1))
                max_var += 1
                clause = clause[1:]
            kb_3sat.add((-max_var, *clause))
        else:
            kb_3sat.add(clause)
    return kb_3sat


def unit_propogate(var, kb, literals):
    '''
    Performs unit propogation of the literal given by var on the knowledge base
    given by kb, thus removing all clauses which are satisfied by assigning the
    literal to its given polarity, and removing the literal in its opposite
    polarity from any clauses which contain the literal in its reversed polarity.

    Creates an empty queue (set) of new unit clauses and adds var to that set.
    Then, while the queue is not empty, pops a literal from the queue and loops
    through each clause in the the kb, removing the clause from the kb if the
    literal occurs in the same polarity in the clause, or replacing the clause
    with a new clause containing all literals besides the complement of the
    literal if that complement occurs in the clause. If removing the complement
    of the literal from the clause results in the clause being empty, then that
    is a contradiction, so returns False. If removing the complement of the
    literal from the clause results in a unit clause, then that unit clause is
    not added to the kb, but is instead added to literals and to the queue of
    new unit clauses. If a new unit clause conflicts with an existing literal,
    then that is a contradiction, so returns False.

    Returns (kb, literals), which is the resulting kb and literals sets after
    completing unit propogation on the given literal, as well as all other
    literals from unit clauses which resulted from unit propogations of previous
    literals propogated during the function.
    '''
    literals = literals.copy()
    units = set()
    units.add(var)
    literals.add(var)
    while len(units) > 0:
        var = units.pop()
        if -var in literals:
            return False
        new_kb = set()
        for clause in kb:
            if len(clause) == 0:
                return False
            if var in clause:
                continue
            if -var in clause:
                remaining = [v for v in clause if v != -var]
                if len(remaining) == 0:
                    return False
                if len(remaining) == 1:
                    new_lit = remaining[0]
                    if -new_lit in literals:
                        return False
                    units.add(new_lit)
                    literals.add(new_lit)
                else:
                    new_kb.add(tuple(remaining))
                continue
            new_kb.add(clause)
        kb = new_kb
    return kb, literals


def pure_literal_assign(kb, literals):
    '''
    Performs pure literal assignment on the given knowledge base by satisfying
    all variables which only occur in one polarity.

    Loops through the knowledge base, recording the polarity of each literal and
    the set of clauses containing that literal in that polarity in a dict of
    pure literals. If a literal occurs in both polarities in the kb, then it is
    removed from the dict of pure literals and added to the set of non-pure
    literals.

    For each literal which remains pure after looping through the complete kb,
    all clauses containing that literal are removed from the kb and the literal
    is added to the literals set, since assigning that literal its given
    polarity does not result in contradictions, and reduces the constraints in
    the kb by satisfying all clauses which contain it.

    Returns (kb, literals), which is the resulting kb and literals sets after
    assigning all pure literals in the knowledge base their corresponding
    values.
    '''
    not_pure = set()
    pure = {}
    for clause in kb:
        for var in clause:
            if abs(var) in not_pure:
                continue
            if -var in pure:
                not_pure.add(abs(var))
                del pure[-var]
                continue
            if var not in pure:
                pure[var] = set()
            pure[var].add(clause)
    new_kb = kb
    for var in pure:
        literals.add(var)
        new_kb = new_kb - pure[var]
    return kb, literals


def dpll_helper(kb, literals):
    '''
    Computes the satisfiability of the given knowledge base using a process
    based on the Davis-Putnam-Logemann-Loveland (DPLL) Algorithm.

    The knowledge base is given by kb, which must be a set containing clauses
    which are tuples of integers, where a positive or negative integer
    corresponds to a value of True of False, respectively, for the associated
    literal. literals is a set of consistent literals which would otherwise be
    unit clauses in kb, but which have since been unit propogated, and thus do
    not occur anywhere in kb. kb must not contain any empty clauses or unit
    clauses.

    The bulk of the DPLL algorithm, as well as the particular optimizations I
    have made, is described in the docstring for the dpll() function.

    Notably, there will never be any inconsistent literal values in literals,
    there will never be any empty clauses in the kb, there will never be any
    unit clauses in the kb, and every literal in literals (or which will be
    added to literals in the future) will have been unit propogated as it is
    added to literals. In fact, the unit_propogate() and pure_literal_assign()
    functions are the only places where values are ever added to the literals
    set. As a result, the only work that this function must do is to call
    pure_literal_assign() (which is optimized to only require one pass through
    the kb), and then choose a variable to try propogating as either True or
    False (short circuiting if the first attempt results in a satisfiable kb).
    The pure_literal_assign() function will never produce additional unit
    clauses, since clauses are either left unchanged if they do not contain a
    pure literal, or they are deleted entirely.

    Returns False if the knowledge base is unsatisfiable, or returns the set of
    literals which resulted in the knowledge base being satisfiable. As before,
    a positive or negative value indicates a True or False assignment,
    respectively, to the given literal.
    '''
    kb, literals = pure_literal_assign(kb, literals)
    if len(kb) == 0:
        # kb is a consistent set of literals, which are stored in literals set
        return literals
    # Now, choose new variable to unit propogate
    var = None
    for clause in kb:
        for v in clause:
            var = v
            break
        break
    result_with_var = unit_propogate(var, kb, literals)
    if result_with_var != False:
        satisfiable_with_var = dpll_helper(*result_with_var)
        if satisfiable_with_var != False:
            return satisfiable_with_var
    # If this is reached, kb was unsatisfiable with var added
    result_with_neg_var = unit_propogate(-var, kb, literals)
    if result_with_neg_var != False:
        return dpll_helper(*result_with_neg_var)
    # Unit propogation of -var result in an unsatisfiable kb
    return False


def dpll(kb):
    '''
    Computes the satisfiability of the given knowledge base using a process
    based on the the Davis-Putnam-Logemann-Loveland (DPLL) Algorithm.

    The knowledge base is given by kb, which must be an iterable containing
    clauses which are tuples of integers, where a positive or negative integer
    corresponds to a value of True or False, respectively, for the associated
    literal.

    The standard DPLL algorithm is as follows.
    DPLL(kb):
        if kb is a consistent set of literals:
            return True
        if kb contains an empty clause:
            return False
        for every unit clause (var) in kb:
            kb = unit_propogate(var, kb)
        for every literal var that only occurs in one polarity in kb:
            kb = pure_literal_assign(var, kb)
        var = choose_literal(kb)
        return dpll(kb + {(var)}) or dpll(kb + {(-var)})

    However, I have made several optimizations to improve runtime. This
    function, dpll(), establishes some properties of kb and literals which
    are leveraged by other functions, most notably dpll_helper().

    1. Rather than repeatedly looping through the kb to identify unit clauses,
    I instead store all unit clauses as the set labeled literals, and they are
    removed from the kb itself.

    2. As soon as a unit clause or literal is identified, it is immediately unit
    propogated, thus removing it from all clauses in the kb (and removing any
    clauses which are satisfied as a result of the literal occurring in that
    polarity).

    3. Rather than allowing inconsistent literals or empty clauses to be added
    to the kb (or to the literals set, in the former case), dpll() and the other
    associated functions instead immediately return False.

    4. Once this function calls dpll_helper(), there will never be any
    inconsistent literal values in literals, there will never be any empty
    clauses in the kb, there will never be any unit clauses in the kb, and
    every literal in literals (or which will be added to literals in the future)
    will have been unit propogated as it is added to literals. In fact, the
    unit_propogate() and pure_literal_assign() functions are the only places
    where values are ever added to the literals set. As a result, the only work
    that dpll_helper() must do on any subsequent call is to call
    pure_literal_assign() (which is optimized to only require one pass through
    the kb), and then choose a variable to try propogating as either True or
    False (short circuiting if the first attempt results in a satisfiable kb).
    The pure_literal_assign() function will never produce additional unit
    clauses, since clauses are either left unchanged if they do not contain a
    pure literal, or they are deleted entirely.

    Returns False if the knowledge base is unsatisfiable, or returns the set of
    literals which resulted in the knowledge base being satisfiable. As before,
    a positive or negative value indicates a True of False assignment,
    respectively, to the associated literal.
    '''
    if len(kb) == 0:
        return set()
    new_kb = set()
    units = set()
    for clause in kb:
        if len(clause) == 0:
            return False
        if len(clause) == 1:
            var = clause[0]
            if -var in units:
                return False
            units.add(var)
        else:
            new_kb.add(clause)
    literals = set()    # set of literals which have been unit propogated
    for unit in sorted(units):
        if unit in literals:
            continue
        if -unit in literals:
            return False
        result = unit_propogate(unit, new_kb, literals)
        if result == False:
            return False
        new_kb, literals = result
    # Now, there are no unit clauses in new_kb, and literals is complete
    # No var in literals occurs in any clause in kb
    return dpll_helper(new_kb, literals)


def kb_satisfiable(kb):
    '''
    Computes whether the knowledge base given by kb is satisfiable by running
    a the dpll() function, which uses a process based on the DPLL algorithm, as
    described in its docstring.

    Returns True if the knowledge base is satisfiable, else False.
    '''
    if dpll(kb) == False:
        return False
    return True


def add_literal_to_kb(var, kb):
    '''
    Creates a new knowledge base consisting of the literal given by var as a
    unit clause, and all clauses from the original knowledge base given by kb.

    Returns the new knowledge base.
    '''
    if type(kb) == type(set()):
        return kb | {(var,)}
    if type(kb) == type([]):
        return kb + [(var,)]
    if type(kb) == type(()):
        return kb + ((var,),)
    # If here, kb is not one of the preferred types
    new_kb = set([(var,)])
    for clause in kb:
        new_kb.add(tuple(clause))
    return new_kb


def test_literal(var, kb):
    '''
    Tests whether the knowledge base given by kb entails that the literal given
    by var must be True or False, or whether the knowledge base does not entail
    any particular polarity for the given literal.

    Adds the given literal to the knowledge base in its given polarity, and if
    the resulting knowledge base is unsatisfiable, returns False. Then adds the
    given literal to the knowledge base in its opposite polarity, and if the
    resulting knowledge base is unsatisfiable, then returns False. If the
    knowledge base is satisfiable with the literal in either polarity, then it
    is unknown whether the kb entails the literal, so returns None.

    Returns True if kb intails var, else False if kb entails -var, else None.
    '''
    if kb_satisfiable(add_literal_to_kb(var, kb)) == False:
        return False
    if kb_satisfiable(add_literal_to_kb(-var, kb)) == False:
        return True
    return None


# BELOW THIS LINE IS TESTING CODE, AS WELL AS THE if __name__ == '__main__': STATEMENT


TRUTHTELLER_CLAUSES = ((-1, 3), (-1, 1), (-3, -1, 1), (-2, -3), (3, 2), (-3, 2, -1), (-2, 3), (1, 3))

CLUE_REASONER_CLAUSES = ((1, 22, 43, 64, 85, 106, 127,), (2, 23, 44, 65, 86, 107, 128,), (3, 24, 45, 66, 87, 108, 129,), (4, 25, 46, 67, 88, 109, 130,), (5, 26, 47, 68, 89, 110, 131,), (6, 27, 48, 69, 90, 111, 132,), (7, 28, 49, 70, 91, 112, 133,), (8, 29, 50, 71, 92, 113, 134,), (9, 30, 51, 72, 93, 114, 135,), (10, 31, 52, 73, 94, 115, 136,), (11, 32, 53, 74, 95, 116, 137,), (12, 33, 54, 75, 96, 117, 138,), (13, 34, 55, 76, 97, 118, 139,), (14, 35, 56, 77, 98, 119, 140,), (15, 36, 57, 78, 99, 120, 141,), (16, 37, 58, 79, 100, 121, 142,), (17, 38, 59, 80, 101, 122, 143,), (18, 39, 60, 81, 102, 123, 144,), (19, 40, 61, 82, 103, 124, 145,), (20, 41, 62, 83, 104, 125, 146,), (21, 42, 63, 84, 105, 126, 147,), (-1, -22,), (-2, -23,), (-3, -24,), (-4, -25,), (-5, -26,), (-6, -27,), (-7, -28,), (-8, -29,), (-9, -30,), (-10, -31,), (-11, -32,), (-12, -33,), (-13, -34,), (-14, -35,), (-15, -36,), (-16, -37,), (-17, -38,), (-18, -39,), (-19, -40,), (-20, -41,), (-21, -42,), (-1, -43,), (-2, -44,), (-3, -45,), (-4, -46,), (-5, -47,), (-6, -48,), (-7, -49,), (-8, -50,), (-9, -51,), (-10, -52,), (-11, -53,), (-12, -54,), (-13, -55,), (-14, -56,), (-15, -57,), (-16, -58,), (-17, -59,), (-18, -60,), (-19, -61,), (-20, -62,), (-21, -63,), (-1, -64,), (-2, -65,), (-3, -66,), (-4, -67,), (-5, -68,), (-6, -69,), (-7, -70,), (-8, -71,), (-9, -72,), (-10, -73,), (-11, -74,), (-12, -75,), (-13, -76,), (-14, -77,), (-15, -78,), (-16, -79,), (-17, -80,), (-18, -81,), (-19, -82,), (-20, -83,), (-21, -84,), (-1, -85,), (-2, -86,), (-3, -87,), (-4, -88,), (-5, -89,), (-6, -90,), (-7, -91,), (-8, -92,), (-9, -93,), (-10, -94,), (-11, -95,), (-12, -96,), (-13, -97,), (-14, -98,), (-15, -99,), (-16, -100,), (-17, -101,), (-18, -102,), (-19, -103,), (-20, -104,), (-21, -105,), (-1, -106,), (-2, -107,), (-3, -108,), (-4, -109,), (-5, -110,), (-6, -111,), (-7, -112,), (-8, -113,), (-9, -114,), (-10, -115,), (-11, -116,), (-12, -117,), (-13, -118,), (-14, -119,), (-15, -120,), (-16, -121,), (-17, -122,), (-18, -123,), (-19, -124,), (-20, -125,), (-21, -126,), (-1, -127,), (-2, -128,), (-3, -129,), (-4, -130,), (-5, -131,), (-6, -132,), (-7, -133,), (-8, -134,), (-9, -135,), (-10, -136,), (-11, -137,), (-12, -138,), (-13, -139,), (-14, -140,), (-15, -141,), (-16, -142,), (-17, -143,), (-18, -144,), (-19, -145,), (-20, -146,), (-21, -147,), (-22, -43,), (-23, -44,), (-24, -45,), (-25, -46,), (-26, -47,), (-27, -48,), (-28, -49,), (-29, -50,), (-30, -51,), (-31, -52,), (-32, -53,), (-33, -54,), (-34, -55,), (-35, -56,), (-36, -57,), (-37, -58,), (-38, -59,), (-39, -60,), (-40, -61,), (-41, -62,), (-42, -63,), (-22, -64,), (-23, -65,), (-24, -66,), (-25, -67,), (-26, -68,), (-27, -69,), (-28, -70,), (-29, -71,), (-30, -72,), (-31, -73,), (-32, -74,), (-33, -75,), (-34, -76,), (-35, -77,), (-36, -78,), (-37, -79,), (-38, -80,), (-39, -81,), (-40, -82,), (-41, -83,), (-42, -84,), (-22, -85,), (-23, -86,), (-24, -87,), (-25, -88,), (-26, -89,), (-27, -90,), (-28, -91,), (-29, -92,), (-30, -93,), (-31, -94,), (-32, -95,), (-33, -96,), (-34, -97,), (-35, -98,), (-36, -99,), (-37, -100,), (-38, -101,), (-39, -102,), (-40, -103,), (-41, -104,), (-42, -105,), (-22, -106,), (-23, -107,), (-24, -108,), (-25, -109,), (-26, -110,), (-27, -111,), (-28, -112,), (-29, -113,), (-30, -114,), (-31, -115,), (-32, -116,), (-33, -117,), (-34, -118,), (-35, -119,), (-36, -120,), (-37, -121,), (-38, -122,), (-39, -123,), (-40, -124,), (-41, -125,), (-42, -126,), (-22, -127,), (-23, -128,), (-24, -129,), (-25, -130,), (-26, -131,), (-27, -132,), (-28, -133,), (-29, -134,), (-30, -135,), (-31, -136,), (-32, -137,), (-33, -138,), (-34, -139,), (-35, -140,), (-36, -141,), (-37, -142,), (-38, -143,), (-39, -144,), (-40, -145,), (-41, -146,), (-42, -147,), (-43, -64,), (-44, -65,), (-45, -66,), (-46, -67,), (-47, -68,), (-48, -69,), (-49, -70,), (-50, -71,), (-51, -72,), (-52, -73,), (-53, -74,), (-54, -75,), (-55, -76,), (-56, -77,), (-57, -78,), (-58, -79,), (-59, -80,), (-60, -81,), (-61, -82,), (-62, -83,), (-63, -84,), (-43, -85,), (-44, -86,), (-45, -87,), (-46, -88,), (-47, -89,), (-48, -90,), (-49, -91,), (-50, -92,), (-51, -93,), (-52, -94,), (-53, -95,), (-54, -96,), (-55, -97,), (-56, -98,), (-57, -99,), (-58, -100,), (-59, -101,), (-60, -102,), (-61, -103,), (-62, -104,), (-63, -105,), (-43, -106,), (-44, -107,), (-45, -108,), (-46, -109,), (-47, -110,), (-48, -111,), (-49, -112,), (-50, -113,), (-51, -114,), (-52, -115,), (-53, -116,), (-54, -117,), (-55, -118,), (-56, -119,), (-57, -120,), (-58, -121,), (-59, -122,), (-60, -123,), (-61, -124,), (-62, -125,), (-63, -126,), (-43, -127,), (-44, -128,), (-45, -129,), (-46, -130,), (-47, -131,), (-48, -132,), (-49, -133,), (-50, -134,), (-51, -135,), (-52, -136,), (-53, -137,), (-54, -138,), (-55, -139,), (-56, -140,), (-57, -141,), (-58, -142,), (-59, -143,), (-60, -144,), (-61, -145,), (-62, -146,), (-63, -147,), (-64, -85,), (-65, -86,), (-66, -87,), (-67, -88,), (-68, -89,), (-69, -90,), (-70, -91,), (-71, -92,), (-72, -93,), (-73, -94,), (-74, -95,), (-75, -96,), (-76, -97,), (-77, -98,), (-78, -99,), (-79, -100,), (-80, -101,), (-81, -102,), (-82, -103,), (-83, -104,), (-84, -105,), (-64, -106,), (-65, -107,), (-66, -108,), (-67, -109,), (-68, -110,), (-69, -111,), (-70, -112,), (-71, -113,), (-72, -114,), (-73, -115,), (-74, -116,), (-75, -117,), (-76, -118,), (-77, -119,), (-78, -120,), (-79, -121,), (-80, -122,), (-81, -123,), (-82, -124,), (-83, -125,), (-84, -126,), (-64, -127,), (-65, -128,), (-66, -129,), (-67, -130,), (-68, -131,), (-69, -132,), (-70, -133,), (-71, -134,), (-72, -135,), (-73, -136,), (-74, -137,), (-75, -138,), (-76, -139,), (-77, -140,), (-78, -141,), (-79, -142,), (-80, -143,), (-81, -144,), (-82, -145,), (-83, -146,), (-84, -147,), (-85, -106,), (-86, -107,), (-87, -108,), (-88, -109,), (-89, -110,), (-90, -111,), (-91, -112,), (-92, -113,), (-93, -114,), (-94, -115,), (-95, -116,), (-96, -117,), (-97, -118,), (-98, -119,), (-99, -120,), (-100, -121,), (-101, -122,), (-102, -123,), (-103, -124,), (-104, -125,), (-105, -126,), (-85, -127,), (-86, -128,), (-87, -129,), (-88, -130,), (-89, -131,), (-90, -132,), (-91, -133,), (-92, -134,), (-93, -135,), (-94, -136,), (-95, -137,), (-96, -138,), (-97, -139,), (-98, -140,), (-99, -141,), (-100, -142,), (-101, -143,), (-102, -144,), (-103, -145,), (-104, -146,), (-105, -147,), (-106, -127,), (-107, -128,), (-108, -129,), (-109, -130,), (-110, -131,), (-111, -132,), (-112, -133,), (-113, -134,), (-114, -135,), (-115, -136,), (-116, -137,), (-117, -138,), (-118, -139,), (-119, -140,), (-120, -141,), (-121, -142,), (-122, -143,), (-123, -144,), (-124, -145,), (-125, -146,), (-126, -147,), (127, 128, 129, 130, 131, 132,), (-127, -128,), (-127, -129,), (-127, -130,), (-127, -131,), (-127, -132,), (-128, -129,), (-128, -130,), (-128, -131,), (-128, -132,), (-129, -130,), (-129, -131,), (-129, -132,), (-130, -131,), (-130, -132,), (-131, -132,), (133, 134, 135, 136, 137, 138,), (-133, -134,), (-133, -135,), (-133, -136,), (-133, -137,), (-133, -138,), (-134, -135,), (-134, -136,), (-134, -137,), (-134, -138,), (-135, -136,), (-135, -137,), (-135, -138,), (-136, -137,), (-136, -138,), (-137, -138,), (139, 140, 141, 142, 143, 144, 145, 146, 147,), (-139, -140,), (-139, -141,), (-139, -142,), (-139, -143,), (-139, -144,), (-139, -145,), (-139, -146,), (-139, -147,), (-140, -141,), (-140, -142,), (-140, -143,), (-140, -144,), (-140, -145,), (-140, -146,), (-140, -147,), (-141, -142,), (-141, -143,), (-141, -144,), (-141, -145,), (-141, -146,), (-141, -147,), (-142, -143,), (-142, -144,), (-142, -145,), (-142, -146,), (-142, -147,), (-143, -144,), (-143, -145,), (-143, -146,), (-143, -147,), (-144, -145,), (-144, -146,), (-144, -147,), (-145, -146,), (-145, -147,), (-146, -147,), (6,), (20,), (21,), (26,), (-46,), (-53,), (-57,), (-67,), (-74,), (-78,), (88, 95, 99,), (-64,), (-72,), (-80,), (85, 93, 101,), (-90,), (-91,), (-101,), (111, 112, 122,), (-108,), (-113,), (-120,), (-3,), (-8,), (-15,), (-24,), (-29,), (-36,), (45, 50, 57,), (6,), (23,), (46, 52, 59,), (64, 71, 84,), (88, 91, 99,), (106, 116, 120,), (-3,), (-7,), (-18,), (-24,), (-28,), (-39,), (45, 49, 60,), (35,), (46, 49, 57,), (67, 75, 76,), (-90,), (-95,), (-102,), (111, 116, 123,), (-110,), (-116,), (-118,), (-5,), (-11,), (-13,), (26, 32, 34,), (-4,), (-11,), (-17,), (-25,), (-32,), (-38,), (-46,), (-53,), (-59,), (-67,), (-74,), (-80,), (-88,), (-95,), (-101,), (109, 130,), (116, 137,), (122, 143,), (-27,), (-32,), (-34,), (-48,), (-53,), (-55,), (-69,), (-74,), (-76,), (97,), (-67,), (-74,), (-76,), (88, 95, 97,), (-109,), (-116,), (-118,), (-4,), (-11,), (-13,), (-25,), (-32,), (-34,), (-46,), (-53,), (-55,), (-67,), (-74,), (-76,), (88, 130,), (95, 137,), (97, 139,), (-24,), (-32,), (-42,), (45,), (-46,), (-53,), (-59,), (-67,), (-74,), (-80,), (-88,), (-95,), (-101,), (109, 116, 122,), (-67,), (-74,), (-84,), (-88,), (-95,), (-105,), (-109,), (-116,), (-126,), (21,), (-90,), (-95,), (-105,), (-111,), (-116,), (-126,), (6,), (-111,), (-116,), (-126,), (6,), (-4,), (-11,), (-16,), (-25,), (-32,), (-37,), (-46,), (-53,), (-58,), (67, 74, 79,),)

CLUE_REASONER_SOLUTION = ((130,), (137,), (145,))


def test_3sat():
    '''
    Run tests to verify that convert_kb_to_3sat() is operating correctly.
    '''
    print('\n\nTESTING 3-SAT CONVERSION:')
    print('-------------------------')

    kb = set(CLUE_REASONER_CLAUSES)
    print(f'\nLoaded Clue KB containing {len(kb)} clauses.')
    print('Converting KB to 3-SAT... ', end='')
    start_time = time.time()
    kb_3sat = convert_kb_to_3sat(kb)
    end_time = time.time()
    print(f'{end_time - start_time} seconds\n')
    print('Clauses in original KB but not in 3-SAT KB:')
    for clause in sorted_kb(set(kb) - kb_3sat):
        print(clause)
    print('\nClauses in 3-SAT KB but not in original KB:')
    for clause in sorted_kb(kb_3sat - set(kb)):
        print(clause)

    print('\nSome simpler examples, for sanity checks by observation:')
    for i in range(1, 10):
        kb = [tuple([k for k in range(1, i+1)])]
        print('\nInitial KB:')
        print(sorted_kb(kb))
        print('3-SAT KB:')
        print(sorted_kb(convert_kb_to_3sat(kb)))

    kb = TRUTHTELLER_CLAUSES
    print('\nOriginal Truthteller KB:')
    print(sorted_kb(kb))
    print('3-SAT Truthteller KB (this should be identical):')
    print(sorted_kb(convert_kb_to_3sat(kb)))

    print('\nCONCLUDED 3-SAT CONVERSION TESTING.')


def test_read_write_kb_to_file():
    '''
    Run tests to verify that reading and writing knowledge bases to files using
    read_kb_from_file(), read_kb_from_fd(), dump_kb_to_file(), and dump_kb_to_fd()
    operates correctly.
    '''
    print('\n\nTESTING READING/WRITING KB TO FILE:')
    print('-----------------------------------')

    kb = CLUE_REASONER_CLAUSES
    tmp_filename = f'/tmp/test_kb_{time.time()}.txt'
    print(f'\nDumping Clue KB to {tmp_filename}')
    dump_kb_to_file(kb, tmp_filename)
    print(f'Reading Clue KB from {tmp_filename}')
    kb_from_file = read_kb_from_file(tmp_filename)
    if sorted_kb(kb) == sorted_kb(kb_from_file):
        print('\nOriginal and written/read KB match.')
    else:
        print('\nWARNING: Original and written/read KB DO NOT match.')

    kb = TRUTHTELLER_CLAUSES
    tmp_filename = f'/tmp/test_kb_{time.time()}.txt'
    print(f'\nDumping Truthteller KB to {tmp_filename}')
    dump_kb_to_file(kb, tmp_filename)
    print(f'Reading Truthteller KB from {tmp_filename}')
    kb_from_file = read_kb_from_file(tmp_filename)
    if sorted_kb(kb) == sorted_kb(kb_from_file):
        print('\nOriginal and written/read KB match.')
    else:
        print('\nWARNING: Original and written/read KB DO NOT match.')

    print('\nCONCLUDED READ/WRITE TESTING.')


def test_sat():
    '''
    Run tests to verify that the satisfiability solver defined in dpll() and its
    associated functions correctly and efficiently computes the satisfiability
    of knowledge bases.
    '''
    print('\n\nTESTING SATISFIABILITY SOLVER:')
    print('------------------------------')

    kb = TRUTHTELLER_CLAUSES
    print('\nTruthteller Clauses:')
    print(sorted_kb(kb))

    print('\nTiming Truthteller checks using DPLL...')
    start_time = time.time()
    people = ['Amy', 'Bob', 'Cal']
    for i in range(1, len(people) + 1):
        start_time = time.time()
        truthful = test_literal(i, kb)
        end_time = time.time()
        print(f'{people[i - 1]} is a truthteller: ', end='')
        print(f'{"Unknown" if truthful is None else truthful} ', end='')
        print(f'in {end_time - start_time} seconds')

    kb = CLUE_REASONER_CLAUSES
    print('\nTiming original Clue KB satisfiability using DPLL...')
    start_time = time.time()
    satisfiable = kb_satisfiable(kb)
    end_time = time.time()
    print(f'{satisfiable} in {end_time - start_time} seconds')
    start_time = time.time()
    satisfiable = kb_satisfiable(kb + CLUE_REASONER_SOLUTION)
    end_time = time.time()
    print(f'{satisfiable} in {end_time - start_time} seconds with solution in clauses')

    kb_3sat = convert_kb_to_3sat(kb)
    print('\nTiming 3-SAT Clue KB satisfiability using DPLL...')
    start_time = time.time()
    satisfiable = kb_satisfiable(kb_3sat)
    end_time = time.time()
    print(f'{satisfiable} in {end_time - start_time} seconds')
    start_time = time.time()
    satisfiable = kb_satisfiable(kb_3sat | set(CLUE_REASONER_SOLUTION))
    end_time = time.time()
    print(f'{satisfiable} in {end_time - start_time} seconds with solution in clauses')

    print('\nCONCLUDED SATISFIABILITY SOLVER TESTING.')


def run_tests():
    '''
    Run tests of the core functionality of this program, including conversion
    of knowledge bases to 3-SAT form, reading and writing knowledge bases to
    files, and correctly and efficiently computing the satisfiability of
    knowledge bases.
    '''
    test_3sat()
    test_read_write_kb_to_file()
    test_sat()


def dump_sample_kbs():
    '''
    Writes the built-in knowledge bases to files.

    Writes the clauses from the Clue reasoner to 'clue_reasoner_clauses.txt'
    Writes the truthteller clauses to 'truthteller_clauses.txt'
    '''
    dump_kb_to_file(CLUE_REASONER_CLAUSES, 'clue_reasoner_clauses.txt')
    dump_kb_to_file(TRUTHTELLER_CLAUSES, 'truthteller_clauses.txt')


def print_test_literal(result_bool):
    '''
    Prints 'TRUE' if the given result is True, prints 'FALSE' if it is False,
    and prints 'UNKNOWN' if the result is None. Designed to print a result from
    a call to test_literal().
    '''
    if result_bool is None:
        print('UNKNOWN')
    elif result_bool == False:
        print('FALSE')
    else:
        print('TRUE')


def print_sat(result_bool):
    '''
    Prints 'SAT' if the given result is True, prints 'UNSAT' if it is False, and
    prints 'UNKNOWN' otherwise. The value of result_bool should always be either
    True or False. Designed to print a result from a call to kb_satisfiable().
    '''
    if result_bool == True:
        print('SAT')
    elif result_bool == False:
        print('UNSAT')
    else:
        print('UNKNOWN')


def print_literals(result):
    '''
    Prints the set of literals which result in the corresponding knowledge base
    being satisfiable. If no assignment of literals can make the knowledge base
    satisfiable, then prints 'UNSAT'. Designed to take a result from a call to
    the dpll() function.
    '''
    if result == False:
        print('UNSAT')
    elif result is None:
        print('UNKNOWN')
    else:
        print(sorted_literals(result))


def main():
    '''
    Parse command line arguments and print the results.
    '''
    parser = argparse.ArgumentParser(description='Compute the satisfiability of knowledge bases in CNF form. See the docstring of the dpll() function for a discussion of the algorithm which is used to compute the satisfiability of knowledge bases.')

    parser.add_argument('kb_file', type=argparse.FileType('r'), nargs='?', help='Load the CNF knowledge base from the given filename (- for stdin).')

    exclusive = parser.add_mutually_exclusive_group()
    exclusive.add_argument('-t', '--test-literal', type=int, help='Compute whether the knowledge base entails the given literal in its given polarity to be True, False, or Unknown. If present, no other arguments are allowed.')
    exclusive.add_argument('-s', '--sat-literal', type=int, nargs='+', help='Compute the satisfiability of the knowledge base if the given literal is added to the knowledge base. May include several consecutive literals to test the satisfiability of the knowledge base when multiple literals are added to the it at once.')
    exclusive.add_argument('--run-tests', action='store_true', help='Run tests on the core functionality of this program.')
    exclusive.add_argument('--dump-sample-kbs', action='store_true', help='Write the built-in clue reasoner clauses and the truthteller clauses to files')
    parser.add_argument('-l', '--print-literals', action='store_true', help='Print out the literal assignments which make the given knowledge base satisfiable, of UNSAT if the knowledge base is unsatisfiable. May be used with no arguments other than --sat-literal.')

    args = parser.parse_args()
    given_args = [arg for arg in vars(args) if arg != 'kb_file' and vars(args)[arg]]
    if args.run_tests:
        if len(given_args) > 1:
            print(f'ERROR: received unexpected arg[s] in addition to --run-tests\n')
            parser.print_help()
            sys.exit(1)
        run_tests()
        sys.exit(0)
    if args.dump_sample_kbs:
        if len(given_args) > 1:
            print(f'ERROR: received unexpected arg[s] in addition to --dump-sample-kbs\n')
            parser.print_help()
            sys.exit(1)
        dump_sample_kbs()
        sys.exit(0)
    if args.kb_file is None:
        parser.print_usage()
        print(f'{sys.argv[0]}: error: the following arguments are required if --run-tests and --dump-sample-kbs are\nnot present: kb_file')
        sys.exit(1)
    kb = read_kb_from_fd(args.kb_file)
    if args.test_literal is not None:
        if args.sat_literal is not None or args.print_literals == True:
            print(f'ERROR: received unexpected arg[s] in addition to -t/--test-literal\n')
            parser.print_help()
            sys.exit(1)
        print_test_literal(test_literal(args.test_literal, kb))
        sys.exit(0)
    if args.sat_literal is not None:
        for literal in args.sat_literal:
            kb = add_literal_to_kb(literal, kb)
    if args.print_literals:
        print_literals(dpll(kb))
    else:
        print_sat(kb_satisfiable(kb))
    sys.exit(0)


if __name__ == '__main__':
    main()
