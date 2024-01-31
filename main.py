# 使用numpy库
import numpy as np
# 使用random库
import random
# 使用keras库
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from flask import Flask, jsonify,request
from flask_cors import CORS
#from fastapi import FastAPI
#import uvicorn

# 定义一些常量
state_size = 19

action_size = 4

batch_size = 32

episodes = 10000

done = False

#SCREEN_WIDTH = 480  # 屏幕宽度
#SCREEN_HEIGHT = 480  # 屏幕高度
#GRID_SIZE = 20  # 网格大小
GRID_WIDTH = 30
GRID_HEIGHT = 30

#input_model_path='/kaggle/input/model4800/model4800'
#html_path='/kaggle'

directionNum = {
    'left': {
        'x': -1,
        'y': 0,
        'rotate': 180 #蛇头在不同的方向中 应该进行旋转
    },
    'right': {
        'x': 1,
        'y': 0,
        'rotate': 0
    },
    'up': {
        'x': 0,
        'y': -1,
        'rotate': -90
    },
    'down': {
        'x': 0,
        'y': 1,
        'rotate': 90
    }
}

# 全局变量
snake= {}
food = {}
state = np.array([])

count = 4801

# 继续定义代理类
class Agent:
    def __init__(self):
        try:
            self.model=load_model(input_model_path)
            self.epsilon = 0.01  # 探索率
            print('load_success!!')
        except:
            # 创建一个神经网络模型，用于近似Q值函数。模型有三个全连接层，输入层有11个神经元（对应状态空间维度），输出层有4个神经元（对应动作空间维度），中间层有24个神经元。使用relu激活函数和adam优化器。
            self.model = Sequential()
            self.model.add(Dense(36, input_dim=state_size, activation='relu'))
            self.model.add(Dense(36, activation='relu'))
            self.model.add(Dense(action_size, activation='linear'))
            self.model.compile(loss='mse', optimizer=Adam(lr=0.001))
            self.epsilon = 1.0  # 探索率
            print('load_fail')
        # 创建一个记忆列表，用于存储经验元组（状态，动作，奖励，下一个状态，是否结束）
        self.memory = []

        # 定义一些超参数
        self.gamma = 0.95  # 折扣因子

        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减因子

    # 根据当前状态选择一个动作，使用ε-贪心策略
    def act(self, state):
        # 以一定的概率随机选择一个动作
        if np.random.rand() <= self.epsilon:
            return random.randrange(action_size)

        # 否则使用模型预测各个动作的Q值，并选择最大的一个
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    # 将经验元组存储到记忆列表中
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 从记忆列表中随机抽取一批经验，并用目标Q值更新模型
    def replay(self, batch_size):
        # 如果记忆列表的长度小于批量大小，就返回
        if len(self.memory) < batch_size:
            return

        # 从记忆列表中随机抽取一批经验
        minibatch = random.sample(self.memory, batch_size)

        # 遍历每个经验元组
        for state, action, reward, next_state, done in minibatch:
            # 如果游戏结束，就将目标Q值设为奖励值
            if done:
                target = reward

            # 否则，根据贝尔曼方程计算目标Q值，即奖励值加上折扣后的下一个状态的最大Q值
            else:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            # 用模型预测当前状态的Q值，并将实际执行的动作对应的Q值替换为目标Q值
            target_q_values = self.model.predict(state)
            target_q_values[0][action] = target

            # 用目标Q值更新模型
            self.model.fit(state, target_q_values, epochs=1, verbose=0)

        # 如果探索率大于最小探索率，就按衰减因子衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



# 定义一个函数，用于获取当前状态。状态包括以下11个特征：
# - 蛇头的横坐标（归一化）
# - 蛇头的纵坐标（归一化）
# - 食物的横坐标（归一化）
# - 食物的纵坐标（归一化）
# - 蛇头是否在食物的左边（布尔值）
# - 蛇头是否在食物的右边（布尔值）
# - 蛇头是否在食物的上边（布尔值）
# - 蛇头是否在食物的下边（布尔值）
# - 蛇头是否朝向墙壁（布尔值）
# - 蛇头是否朝向自己身体（布尔值）
# - 蛇的长度（归一化）

def get_state():
    # 获取蛇头和食物的位置
    head_x = snake['positions'][0][0]
    head_y = snake['positions'][0][1]
    food_x = food['positions'][0]
    food_y = food['positions'][1]

    # 计算蛇头和食物的相对位置关系
    left_of_food = head_x < food_x
    right_of_food = head_x > food_x
    above_food = head_y < food_y
    below_food = head_y > food_y

    # 方向
    direction_LEFT = (snake['direction'] == directionNum['left'])
    direction_RIGHT = (snake['direction'] == directionNum['right'])
    direction_UP = (snake['direction'] == directionNum['up'])
    direction_DOWN = (snake['direction'] == directionNum['down'])

    # 计算蛇头是否朝向墙壁或者自己身体
    facing_wall = (head_x == 0 and direction_LEFT) or \
                  (head_x == GRID_WIDTH-1 and direction_RIGHT) or \
                  (head_y == 0 and direction_UP) or \
                  (head_y == GRID_HEIGHT-1 and direction_DOWN)

    facing_body = ((head_x, head_y) in snake['positions'][1:] and direction_LEFT) or \
                  ((head_x, head_y) in snake['positions'][1:] and direction_RIGHT) or \
                  ((head_x, head_y) in snake['positions'][1:] and direction_UP) or \
                  ((head_x, head_y) in snake['positions'][1:] and direction_DOWN)

    # 身体方位
    left_to_body=0
    right_to_body=0
    up_to_body=0
    down_to_body=0
    for pos in snake['positions'][1:]:
        if pos[1] == snake['positions'][0][1] and pos[0] < snake['positions'][0][0]:
            left_to_body = 1
        if pos[1] == snake['positions'][0][1] and pos[0] > snake['positions'][0][0]:
            right_to_body = 1
        if pos[0] == snake['positions'][0][0] and pos[1] > snake['positions'][0][1]:
            up_to_body = 1
        if pos[0] == snake['positions'][0][0] and pos[1] < snake['positions'][0][1]:
            down_to_body = 1
    # 获取蛇的长度
    length = snake['length']

    # 将各个特征归一化并转换为numpy数组，作为状态返回
    state = np.array([head_x / GRID_WIDTH,
                      head_y / GRID_HEIGHT,
                      food_x / GRID_WIDTH,
                      food_y / GRID_HEIGHT,
                      int(direction_LEFT),
                      int(direction_RIGHT),
                      int(direction_UP),
                      int(direction_DOWN),
                      left_to_body,
                      right_to_body,
                      up_to_body,
                      down_to_body,
                      int(left_of_food),
                      int(right_of_food),
                      int(above_food),
                      int(below_food),
                      int(facing_wall),
                      int(facing_body),
                      length / (GRID_WIDTH * GRID_HEIGHT)])

    return state.reshape(1, state_size)


def get_distance():
    return abs(snake['positions'][0][0]-food['positions'][0])+abs(snake['positions'][0][1]-food['positions'][1])

# 定义一个函数，用于将动作编号转换为方向
def get_action_key(action):
    # 如果动作编号为0，返回上方向键
    if action == 0:
        return 'up'

    # 如果动作编号为1，返回下方向键
    elif action == 1:
        return 'down'

    # 如果动作编号为2，返回左方向键
    elif action == 2:
        return 'left'

    # 如果动作编号为3，返回右方向键
    elif action == 3:
        return 'right'


# 定义一个函数，用于获取当前奖励。奖励有以下三种情况：
# - 如果蛇吃到食物，奖励为20分
# - 如果蛇死亡，奖励为-10分
# - 否则，奖励为-0.1分
arg_distance=12
# 继续定义get_reward函数
def get_reward():
    global count
    reward=0
    # 如果蛇吃到食物，奖励为10分，并让蛇增加长度，食物重新生成位置
    if snake['positions'][0] == food['positions']:
        reward = 30
    # 如果蛇死亡，奖励为-10分，并设置游戏结束标志为True
    elif snake['is_dead']:
        reward = -15
        if count%100==0:
            save_model(agent.model,"model"+str(count))
        count+=1
    # 否则，奖励为-0.1分，表示惩罚蛇的无效移动
    else:
        reward = -0.1
    if(get_distance() != 0):
        reward+=(arg_distance/get_distance()**2)
    # 返回奖励值
    return reward


agent = Agent()
# 定义一个主函数，用于运行强化学习算法
app = Flask(__name__)
CORS(app)

@app.route('/train', methods=['POST'])
def train():
    global snake,food,state
    data = request.get_json()
    old_state=state.copy()
    snake =data['snake']
    food =data['food']
    state=get_state()
    if(snake!= {} and food != {}):
        # 获取旧的snake和foodstate
        reward=get_reward()
        agent.remember(old_state,data['action'],reward,state,snake['is_dead'])
    if(snake['is_dead']):
        agent.replay(batch_size)
        snake={}
        food= {}
        return jsonify({'action': -1,'direction': ''})
    action=agent.act(state)
    print(action)
    return jsonify({'action': int(action),'direction': get_action_key(action)})

@app.route('/', methods=['POST','GET'])
def hello():
    print('hello')
    page=open('index.html',encoding='utf-8');#打开当前文件下的my_index.html(这个html是你自己写的)
    res=page.read()
    return res

# 运行主函数
if __name__ == "__main__":
    #uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
    app.run(host='0.0.0.0',port=8000,debug=True)
