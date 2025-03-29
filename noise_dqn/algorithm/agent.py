#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :back_to_the_realm
@File    :agent.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""

# import torch
# from kaiwu_agent.agent.base_agent import (
#     predict_wrapper,
#     exploit_wrapper,
#     learn_wrapper,
#     save_model_wrapper,
#     load_model_wrapper,
#     BaseAgent,
# )
# from diy.model.model import Model
# from kaiwu_agent.utils.common_func import attached
# from diy.config import Config


# @attached
# class Agent(BaseAgent):
#     def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
#         super().__init__(agent_type, device, logger, monitor)

#     @predict_wrapper
#     def predict(self, list_obs_data):
#         pass

#     @exploit_wrapper
#     def exploit(self, list_obs_data):
#         pass

#     @learn_wrapper
#     def learn(self, list_sample_data):
#         pass

#     @save_model_wrapper
#     def save_model(self, path=None, id="1"):
#         pass

#     @load_model_wrapper
#     def load_model(self, path=None, id="1"):
#         pass




import torch
import os
import time
import math
import numpy as np
from noise_dqn.model.model import Model, NoisyLinear
from noise_dqn.feature.definition import ActData
from kaiwu_agent.agent.base_agent import (
    BaseAgent, predict_wrapper, exploit_wrapper, 
    learn_wrapper, save_model_wrapper, load_model_wrapper
)
from kaiwu_agent.utils.common_func import attached
from noise_dqn.config import Config

@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.act_shape = Config.DIM_OF_ACTION_DIRECTION + Config.DIM_OF_TALENT
        self.direction_space = Config.DIM_OF_ACTION_DIRECTION
        self.talent_direction = Config.DIM_OF_TALENT
        self.obs_shape = Config.DIM_OF_OBSERVATION
        self._gamma = Config.GAMMA
        self.lr = Config.START_LR

        self.hybrid_epsilon = Config.HYBRID_EPSILON  # 新增混合探索参数

        self.device = device
        self.model = Model(
            state_shape=(Config.DESC_OBS_SPLIT[0],),  # 向量特征维度
            action_shape=(self.act_shape,),
            softmax=False,
        )
        self.model.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_step = 0
        self.last_report_monitor_time = 0

        self.agent_type = agent_type
        self.logger = logger
        self.monitor = monitor

    def __convert_to_tensor(self, data):
        if isinstance(data, list):
            return torch.tensor(np.array(data), device=self.device, dtype=torch.float32)
        else:
            return torch.tensor(data, device=self.device, dtype=torch.float32)

    def __reset_noise(self):
        def reset_noise(module):
            if isinstance(module, NoisyLinear):
                module.reset_noise()
        self.model.apply(reset_noise)

    def __predict_detail(self, list_obs_data, exploit_flag=False):
        batch = len(list_obs_data)
        feature_vec = [obs_data.feature[:Config.DESC_OBS_SPLIT[0]] for obs_data in list_obs_data]
        feature_map = [obs_data.feature[Config.DESC_OBS_SPLIT[0]:] for obs_data in list_obs_data]
        legal_act = [obs_data.legal_act for obs_data in list_obs_data]
        legal_act = torch.tensor(np.array(legal_act))
        legal_act = (
            torch.cat((
                legal_act[:,0].unsqueeze(1).expand(batch, self.direction_space),
                legal_act[:,1].unsqueeze(1).expand(batch, self.talent_direction)
            ), 1).bool().to(self.device)
        )

        model = self.model
        model.train() if not exploit_flag else model.eval()
        if not exploit_flag:
            self.__reset_noise()  # 确保每次预测使用新的噪声样本

        with torch.no_grad():
            # 混合探索策略
            if not exploit_flag and np.random.rand() < self.hybrid_epsilon:
                # 生成合法随机动作
                random_action = np.random.rand(batch, self.act_shape)
                random_action = torch.tensor(random_action, dtype=torch.float32).to(self.device)
                random_action = random_action.masked_fill(~legal_act, 0)
                act = random_action.argmax(dim=1).cpu().view(-1, 1).tolist()
            else:
                # 噪声网络预测
                feature = [
                    self.__convert_to_tensor(feature_vec),
                    self.__convert_to_tensor(feature_map).view(batch, *Config.DESC_OBS_SPLIT[1])
                ]
                logits, _ = model(feature, state=None)
                logits = logits.masked_fill(~legal_act, float(torch.min(logits)))
                act = logits.argmax(dim=1).cpu().view(-1,1).tolist()
            # feature = [
            #     self.__convert_to_tensor(feature_vec),
            #     self.__convert_to_tensor(feature_map).view(batch, *Config.DESC_OBS_SPLIT[1])
            # ]
            # logits, _ = model(feature, state=None)
            # logits = logits.masked_fill(~legal_act, float(torch.min(logits)))
            # act = logits.argmax(dim=1).cpu().view(-1,1).tolist()

        format_action = [[instance[0]%self.direction_space, instance[0]//self.direction_space] 
                        for instance in act]
        return [ActData(move_dir=i[0], use_talent=i[1]) for i in format_action]

    @predict_wrapper
    def predict(self, list_obs_data):
        return self.__predict_detail(list_obs_data, exploit_flag=False)

    @exploit_wrapper
    def exploit(self, list_obs_data):
        return self.__predict_detail(list_obs_data, exploit_flag=True)

    @learn_wrapper
    def learn(self, list_sample_data):
        t_data = list_sample_data
        batch = len(t_data)

        # 数据准备
        batch_feature_vec = [frame.obs[:Config.DESC_OBS_SPLIT[0]] for frame in t_data]
        batch_feature_map = [frame.obs[Config.DESC_OBS_SPLIT[0]:] for frame in t_data]
        batch_action = torch.LongTensor([int(frame.act) for frame in t_data]).view(-1,1).to(self.device)

        _batch_obs_legal = torch.tensor(np.array([frame._obs_legal for frame in t_data]))
        _batch_obs_legal = (
            torch.cat((
                _batch_obs_legal[:,0].unsqueeze(1).expand(batch, self.direction_space),
                _batch_obs_legal[:,1].unsqueeze(1).expand(batch, self.talent_direction)
            ), 1).bool().to(self.device)
        )

        rew = torch.tensor(np.array([frame.rew for frame in t_data]), device=self.device)
        _batch_feature_vec = [frame._obs[:Config.DESC_OBS_SPLIT[0]] for frame in t_data]
        _batch_feature_map = [frame._obs[Config.DESC_OBS_SPLIT[0]:] for frame in t_data]
        not_done = torch.tensor(np.array([0 if frame.done==1 else 1 for frame in t_data]), device=self.device)

        batch_feature = [
            self.__convert_to_tensor(batch_feature_vec),
            self.__convert_to_tensor(batch_feature_map).view(batch, *Config.DESC_OBS_SPLIT[1])
        ]
        _batch_feature = [
            self.__convert_to_tensor(_batch_feature_vec),
            self.__convert_to_tensor(_batch_feature_map).view(batch, *Config.DESC_OBS_SPLIT[1])
        ]

        # 重置噪声
        self.__reset_noise()

        # 计算目标Q值
        self.model.eval()
        with torch.no_grad():
            q, h = self.model(_batch_feature, state=None)
            q = q.masked_fill(~_batch_obs_legal, float(torch.min(q)))
            q_max = q.max(dim=1).values.detach()
            target_q = rew + self._gamma * q_max * not_done

        # 训练步骤
        self.optim.zero_grad()
        self.model.train()
        logits, h = self.model(batch_feature, state=None)

        loss = torch.square(target_q - logits.gather(1, batch_action).view(-1)).mean()
        loss.backward()

        model_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()

        # 监控和日志
        self.train_step += 1
        value_loss = loss.detach().item()
        q_value = target_q.mean().detach().item()
        reward = rew.mean().detach().item()

        now = time.time()
        if now - self.last_report_monitor_time >= 60 and self.monitor:
            self.monitor.put_data({
                os.getpid(): {
                    "value_loss": value_loss,
                    "q_value": q_value,
                    "reward": reward,
                    "diy_1": model_grad_norm,
                    #"diy_2": 0, 
                    "diy_2": self.hybrid_epsilon,  # 在监控数据中跟踪探索率变化
                    "diy_3":0, "diy_4":0, "diy_5":0
                }
            })
            self.last_report_monitor_time = now

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(model_state_dict_cpu, model_file_path)
        self.logger.info(f"Save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        self.logger.info(f"Load model {model_file_path} successfully")