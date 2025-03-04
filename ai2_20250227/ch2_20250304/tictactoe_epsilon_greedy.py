import time
import random

import pytest
import numpy as np


EPS = 0.1
ALPHA = 0.1
MAX_EPISODE = 100000
SHOW_LEARN = False

# side == 1 ==> 系統下棋, 2：User 下棋
# wside == 1 ==> X贏
# wside == 2 ==> O贏
# wside == -1 ==> 平手(draw)

@pytest.fixture
def g():
    return Game()


class IllegalPosition(Exception):
    pass


class Game(object):
    """Tictactoe game class.

    Game state is a numpy array. In the beggining it is filled with 0.
    State side number & symbol
        1: X (CPU)
        2: O (CPU or Player)

    Position is for tile placment, which starts from 1 and ends in 9.

        1|2|3
        -----
        4|5|6
        -----
        7|8|9

    Game state index starts from 0 and ends in 8.

    """

    def __init__(self):
        self.reset_state()
        # state value function
        self.st_values = {}

    def reset_state(self):
        # 記錄目前的棋局
        self.state = np.zeros(9)

    # 判斷目前棋局誰贏誰輸
    """
    這段程式碼是 `Game` 類中的 `query_state_value` 方法，用於查詢遊戲狀態的價值。

以下是程式碼的解釋：

1. `state = self.state if _state is None else _state`：檢查 `_state` 參數是否為空。如果是空，則將 `state` 設置為遊戲物件 `self` 的狀態（`self.state`），否則將 `state` 設置為傳入的 `_state`。

2. `tstate = tuple(state)`：將 `state` 轉換為元組 `tstate`。這是為了確保 `state` 可以作為字典的鍵值使用，因為字典需要不可變的鍵。

3. `if tstate not in self.st_values:`：檢查 `tstate` 是否存在於 `self.st_values` 字典中。如果不存在，表示該遊戲狀態的價值尚未計算。

4. `self.st_values[tstate] = calc_state_value(state)`：如果 `tstate` 不在 `self.st_values` 中，則調用 `calc_state_value` 函數計算該遊戲狀態的價值，並將結果存儲在 `self.st_values` 字典中，以便以後的查詢使用。

5. `return self.st_values[tstate]`：返回 `self.st_values` 字典中 `tstate` 對應的值，即該遊戲狀態的價值。

總結來說，`query_state_value` 方法根據遊戲的當前狀態來查詢該狀態的價值。如果該狀態的價值已經計算過，則直接返回；如果尚未計算，則調用 `calc_state_value` 函數來計算該狀態的價值並存儲起來。這樣可以提高後續的查詢效率，避免重複計算。
    """
    def query_state_value(self, _state=None):
        state = self.state if _state is None else _state
        tstate = tuple(state)
        if tstate not in self.st_values:
            # 判斷誰贏
            # return 0：X 贏
            # return 1：O 贏
            # return 0.5：無輸贏
            # 更新 state value function
            self.st_values[tstate] = calc_state_value(state)
        return self.st_values[tstate]
    """
    get_legal_index 是 Game 類的一個方法，用於獲取當前遊戲狀態下的合法動作位置。

    在井字遊戲（Tic-Tac-Toe）中，每個位置可以是空的、被系統（X）佔據或被使用者（O）佔據。當一個位置是空的時候，它被認為是一個合法的動作位置，即可以在該位置下棋。

    get_legal_index 方法遍歷遊戲狀態的每個位置，檢查該位置是否為空。如果是空的，則將該位置的索引加入到一個列表中。最終，該方法返回包含所有合法動作位置索引的列表。

    例如，如果 get_legal_index 返回 [0, 4, 6, 8]，則表示在當前的遊戲狀態下，可以在棋盤的左上角、中央、右上角和右下角這些位置進行下棋。

    這個方法可以用於獲取當前可以執行的合法動作位置，以供後續的策略選擇和下棋使用。
    """
    def get_legal_index(self):
        return np.nonzero(np.equal(0, self.state))[0].tolist()

    def get_user_input(self, test_input=None):
        """Get user input.

        Args:
            test_input: Test input

        Returns:
            int: Index of input position.

        Raises:
            IllegalPosition
        """
        idx = None
        if test_input is not None:
            inp = test_input
        else:
            inp = input(self.user_input_prompt())
        try:
            idx = int(inp) - 1
        except ValueError:
            raise IllegalPosition()

        if idx < 0 or idx > 8:
            raise IllegalPosition()

        if idx not in self.get_legal_index():
            raise IllegalPosition()

        return idx

    def user_input_prompt(self):
        return "Enter position[1-9]: "

    def draw(self):
        return draw(self.state)


def egreedy_index(state, legal_indices, query_state_value, side, eps=EPS):
    # 若 < epsilon，隨機選一個
    # 因為eps 隨訓練久會越來越小 random.random() < eps 機率就變高
    # legal_indices 當前還可落子的格子
    # 
    if random.random() < eps:
        return random.choices(legal_indices)[0]
    else:
        indices = []
        max_val = -1
        
        # 換人下棋，一方下一次
        # 如果 side == 1 ==> [1, 2] ==> 系統先下，User 再下
        # 如果 side == 2 ==> [2, 1] ==> User 再下，系統先下
       
        for s in [side, 3 - side]:
            for li in legal_indices:
                state[li] = s
                 # 查詢勝率最大的可能性
                val = query_state_value(state)
                # User下棋，val 設為負值，不考慮最大值
                """
                如果當前迭代的玩家是 User（1），
                則 val 的值被設置為 1 減去原始值 val。
                這是因為 User（1）的價值被視為相反的值，
                即 1 - val。這樣做是因為 epsilon-greedy 策略中，在選擇動作時，我們希望系統（X）選擇具有較大價值的動作，而 User（O）選擇具有較小價值的動作。
                """
                if s == 1:
                    val = 1 - val
                    
                # 將最大值的位置存至 indices LIST
                if val > max_val:
                    indices = [li]
                    max_val = val
                    """
                    在這段程式碼中，val < 1.0 的條件是用來過濾價值小於 1.0 的情況。這是因為在遊戲中，如果某個位置的價值為 1.0，表示該位置是一個必勝的位置，不需要進一步考慮其他位置。
                    在 epsilon-greedy 策略中，我們希望選擇具有最大價值的動作，但同時也要保留一定的隨機性，以便探索其他可能性。因此，如果有多個位置具有相同的最大價值，但其中某些位置的價值已達到 1.0，這些位置就不再需要進一步考慮。
                        因此，val < 1.0 的條件確保只有價值小於 1.0 的位置才會被添加到 indices 列表中，從而排除了價值為 1.0 的位置。這樣可以保證選擇的位置是最大價值中的非必勝位置，從而增加探索其他可能性的機會
                    """    
                elif val == max_val and val < 1.0:
                    indices.append(li)
                state[li] = 0 #每次把下棋子歸0
        # 若有多個最大值，隨機選一個
        return random.choices(indices)[0]

# 畫棋盤
def draw(state):
    rv = '\n'
    for y in range(3):
        for x in range(3):
            idx = y * 3 + x
            t = state[idx]
            if t == 1:
                rv += 'X'
            elif t == 2:
                rv += 'O'
            else:
                if x < 2:
                    rv += ' '
            if x < 2:
                rv += '|'
        rv += '\n'
        if y < 2:
            rv += '-----\n'
    return rv


# 判斷誰贏
def judge(g):
    # wside == 1 ==> X贏
    # wside == 2 ==> O贏
    # wside == -1 ==> 平手(draw)
    wside = get_win_side(g.state)
    finish = False
    if wside > 0: # 已有輸贏
        print(g.draw())
        print_winner(wside)
        finish = True
    elif len(g.get_legal_index()) == 0: # 已填滿，平手
        print("Draw!")
        finish = True

    if finish:
        again = input("Play again? (y/n): ")
        if again.lower() != 'y':
            return True
        else:
            g.reset_state()
            return False


def play_turn(g, side):
    # side == 1 ==> 系統下棋
    if side == 1:
        idx = egreedy_index(g.state, g.get_legal_index(), g.query_state_value,
                            side, 0)
    else: # user 下棋
        while True:
            try:
                idx = g.get_user_input()
            except IllegalPosition:
                print("Illegal position!")
            else:
                break

    g.state[idx] = side
    print(g.draw())

    stop = judge(g)
    return 3 - side, stop


def play(_g=None):
    g = Game() if _g is None else _g
    # side == 1 ==> 系統先下棋
    side = 1
    while True:
        side, stop = play_turn(g, side)
        if stop is not None:
            if stop:
                break
            else:
                if side == 2:
                    print(g.draw())
                continue


def print_winner(wside):
    print("Winner is '{}'".format('O' if wside == 2 else 'X'))


def learn():
    g = Game()
    # side == 1 ==> 系統先下棋
    side = 1
    # wside == 1 ==> X贏
    # wside == 2 ==> O贏
    # wside == -1 ==> 平手(draw)
    wside = 0 
    
    # michael add to load result.txt into g.st_values
    import os
    if os.path.exists("result.txt"):
        load_file(g)
    else:
        # learn 100000 次
        for e in range(MAX_EPISODE):
            lidx = g.get_legal_index()
            if len(lidx) == 0:
                wside = -1
            else:
                """
                eps = np.exp(-e*0.0005)：eps 是一個隨著學習進行而變動的變數，通常用於實現 epsilon-greedy 策略，該策略是一種在探索（選擇隨機行動）和利用（選擇當前看起來最好的行動）之間取得平衡的方法。eps
                  的值隨著學習的進行逐漸減小（因為 e，代表當前學習輪數的變數，逐漸增加），這意味著隨著學習的進行，模型越來越傾向於選擇當前看起來最好的行動，而不是隨機行動。

                  在給定的程式碼中，eps = np.exp(-e*0.0005) 是用來計算 ε 的值。它使用指數函數 np.exp 
                  來計算指數衰減的 ε。當 e 增加時，ε 的值會隨之衰減。指數函數的指數部分 -e*0.0005 可以控制衰減的速度。
                  較大的 e 將導致更小的 ε 值，從而增加利用的機會；較小的 e 將導致較大的 ε 值，從而增加探索的機會。
                """
                eps = np.exp(-e*0.0005)
                wside, side = _learn_body(g, lidx, side, eps)

            if SHOW_LEARN:
                time.sleep(1)
                if wside > 0:
                    print_winner(wside)
                elif wside == -1:
                    print('Draw')
            #只要贏了或平手就將棋盤重製
            if wside > 0 or wside == -1:
                g.reset_state()
                if SHOW_LEARN:
                    time.sleep(1)
# (2.0, 0.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0) 
                '''
                O |   | X
                ---------
                X | O | X
                ---------
                X | O | O
                '''
        # key是棋盤       
        save(g.st_values)
        
    g.reset_state()
    return g

# 訓練主體過程
# lidx 還可下子的位置
# side 是誰在下棋
def _learn_body(g, lidx, side, eps):
    state = tuple(g.state)
    # 依 epsilon greedy policy 決定下在 idx 位置
    # 使用 epsilon-greedy 策略來決定在哪個位置下棋。egreedy_index 函數會返回選擇的位置的索引。
    # g.state 目前狀態    
    idx = egreedy_index(g.state, lidx, g.query_state_value, side, eps)
    # get state value
    # 查詢當前遊戲狀態的價值，並將其儲存到變數 value 中。
    value = g.query_state_value()
    # 下在 side 位置
    g.state[idx] = side
    if SHOW_LEARN:
        print(g.draw())


    # get NEW state value
    nvalue = g.query_state_value()
    # update state vlaue
    # 使用 update_values 函數更新原來遊戲狀態的價值。這個函數的具體行為會根據所使用的學習算法而變化。
    g.st_values[state] = update_values(value, nvalue)
    # 判斷是否連成一線
    # 0：否
    # 1：X 贏
    # 2：O 贏
    wside = get_win_side(g.state)
    # 換人下棋 side = 0, 3 
    side = 3 - side
    return wside, side # 最後，返回勝者和下一個玩家。

# update state vlaue
def update_values(this_value, next_value):
    diff = next_value - this_value
    return this_value + ALPHA * diff

# return 0：X 贏
# return 1：O 贏
# return 0.5：無輸贏
def calc_state_value(state):
    # 判斷是否連成一線
    # 0：否
    # 1：X 贏
    # 2：O 贏
    ws = get_win_side(state)
    if ws == 2:
        return 1
    elif ws == 1:
        return 0
    else:
        return 0.5


# 歷次結果存檔
def save(st_values):
    with open("result.txt", "w") as f:
        for state, value in st_values.items():
            f.write("{}: {}\n".format(state, value))

# 歷次訓練結果載入
def load_file(g):
    g.st_values={}
    with open("result.txt", "r") as f:
        list1 = f.readlines()
        for row in list1:
            #print(row)
            state, value = row.split(': ',1)
            g.st_values[state] = float(value)


# 判斷是否連成一線
# 0：否
# 1：X 贏
# 2：O 贏
def get_win_side(_state):
    state = _state.copy().reshape((3, 3))

    for s in [1, 2]:
        for t in range(2):
            for r in range(3):
                if np.array_equal(np.unique(state[r]), [s]):#檢查棋盤的一行（或列）是否全被同一玩家佔據。
                    return s
            state = state.transpose()

        # check diagonals
        if _state[0] == s and _state[4] == s and _state[8] == s:
            return s
        if _state[2] == s and _state[4] == s and _state[6] == s:
            return s

    return 0

'''
    以下 test_* 是 TDD
'''

def test_draw(g):
    assert g.draw() == '''
 | |
-----
 | |
-----
 | |
'''
    assert len(g.state) == 9

    # 1|2|3
    # -----
    # 4|5|6
    # -----
    # 7|8|9
    #
    # X: 1, O: 2
    pos = 1
    idx = pos - 1
    g.state[idx] = 2
    assert g.draw() == '''
O| |
-----
 | |
-----
 | |
'''
    pos = 9
    idx = pos - 1
    g.state[idx] = 1
    assert g.draw() == '''
O| |
-----
 | |
-----
 | |X
'''


def test_user_input(g):
    assert g.user_input_prompt() == "Enter position[1-9]: "
    with pytest.raises(IllegalPosition):
        g.get_user_input('eueueu')

    with pytest.raises(IllegalPosition):
        g.get_user_input('0')

    with pytest.raises(IllegalPosition):
        g.get_user_input('10')

    assert g.get_user_input('1') == 0

    g.state[0] = 1
    with pytest.raises(IllegalPosition):
        g.get_user_input('1')


def test_legal_positions(g):
    assert g.get_legal_index() == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    g.state[0] = 1
    assert g.get_legal_index() == [1, 2, 3, 4, 5, 6, 7, 8]
    g.state[8] = 2
    assert g.get_legal_index() == [1, 2, 3, 4, 5, 6, 7]
    g.state[5] = 1
    assert g.get_legal_index() == [1, 2, 3, 4, 6, 7]


def test_play_update(g):
    g.state = np.array((2, 1, 2,
                        1, 2, 1,
                        0, 0, 0))
    assert 0.5 == g.query_state_value()

    g.state[8] = 2
    assert 1.0 == g.query_state_value()
    assert g.st_values[tuple(g.state)] == 1.0

def test_win_state(g):
    g.state[0] = 1
    g.state[1] = 1
    g.state[2] = 1
    assert 1 == get_win_side(g.state)

    g.reset_state()
    g.state[0] = 1
    g.state[3] = 1
    g.state[6] = 1
    assert 1 == get_win_side(g.state)

    g.reset_state()
    g.state[0] = 2
    g.state[1] = 2
    g.state[2] = 2
    assert 2 == get_win_side(g.state)

    g.reset_state()
    g.state[0] = 2
    g.state[3] = 2
    g.state[6] = 2
    assert 2 == get_win_side(g.state)

    g.reset_state()
    g.state[0] = 1
    g.state[4] = 1
    g.state[8] = 1
    assert 1 == get_win_side(g.state)

    g.reset_state()
    g.state[0] = 2
    g.state[4] = 2
    g.state[8] = 2
    assert 2 == get_win_side(g.state)

    g.reset_state()
    g.state[2] = 1
    g.state[4] = 1
    g.state[6] = 1
    assert 1 == get_win_side(g.state)

    g.reset_state()
    g.state[2] = 2
    g.state[4] = 2
    g.state[6] = 2
    assert 2 == get_win_side(g.state)


def test_state_value(g):
    state = np.zeros(9)
    state[0] = 2
    state[1] = 2
    state[2] = 2
    assert 1 == calc_state_value(state)
    assert 1 == g.query_state_value(state)
    assert 1 == g.st_values[tuple(state)]
    state = np.zeros(9)
    state[1] = 1
    state[1] = 1
    state[2] = 1
    assert 0 == calc_state_value(state)
    assert 0 == g.query_state_value(state)

    state = np.zeros(9)
    assert 0.5 == calc_state_value(state)
    assert 0.5 == g.query_state_value(state)

    assert 1.0 == g.query_state_value(np.array((1.0, 1.0, 2.0, 1.0, 2.0, 1.0,
                                                2.0, 0.0, 2.0)))


def test_egreedy_policy(g):
    """
O|O|   legal index: 2, 7
-----
O|X|X
-----
X| |X

O|O|O  value: 1
-----
O|X|X
-----
X| |X

O|O|   value: 0.5
-----
O|X|X
-----
X|O|X
    """
    state = np.array([2, 2, 0, 2, 1, 1, 1, 0, 1])

    results = []
    g.state = np.array(state)
    for i in range(100):
        res = egreedy_index(state, [2, 7], g.query_state_value, 2)
        results.append(res)

    results = np.array(results)

    gcnt = np.count_nonzero(results == 2)
    rcnt = np.count_nonzero(results == 7)
    assert gcnt >= rcnt * 9


def test_egreedy_policy2(g):
    """
O|O|   legal index: 2, 3, 6, 7, 8
-----
 |X|X
-----
 | |
    """
    state = np.array([2, 2, 0, 0, 1, 1, 0, 0, 0])
    results = []
    g.state = np.array(state)
    for i in range(100):
        res = egreedy_index(state, [2, 3, 6, 7, 8], g.query_state_value, 1)
        results.append(res)

    results = np.array(results)

    gcnt = np.count_nonzero(results == 3)
    rcnt = np.count_nonzero(results != 3)
    assert gcnt >= rcnt * 9


if __name__ == "__main__":
    game = learn()
    play(game)
