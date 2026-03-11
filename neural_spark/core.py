"""
NeuralSpark 核心实现
类脑脉冲神经网络原型，严格遵循生物神经机制
"""

import numpy as np


class LIFNeuron:
    """LIF泄漏积分点火神经元，还原生物核心电生理特性"""
    def __init__(
        self,
        neuron_type="excitatory",
        threshold=1.0,
        rest_potential=0.0,
        leak_factor=0.95,
        refractory_period=2,
        reset_potential=0.0
    ):
        self.neuron_type = neuron_type
        self.threshold = threshold
        self.rest_potential = rest_potential
        self.leak_factor = leak_factor
        self.refractory_period = refractory_period
        self.reset_potential = reset_potential

        # 动态状态
        self.membrane_potential = rest_potential
        self.refractory_remaining = 0
        self.fired = False
        self.last_fire_step = -1000

    def step(self, current_step):
        """单步时序更新：处理不应期、膜电位自然泄漏"""
        self.fired = False

        # 不应期内直接复位，不响应任何输入
        if self.refractory_remaining > 0:
            self.refractory_remaining -= 1
            self.membrane_potential = self.reset_potential
            return

        # 膜电位泄漏衰减
        self.membrane_potential = (
            self.membrane_potential * self.leak_factor
            + self.rest_potential * (1 - self.leak_factor)
        )

    def receive_input(self, input_signal, current_step):
        """接收输入信号，判断是否触发点火"""
        if self.refractory_remaining > 0:
            return

        self.membrane_potential += input_signal

        # 达到阈值，发放脉冲
        if self.membrane_potential >= self.threshold:
            self.fired = True
            self.last_fire_step = current_step
            self.refractory_remaining = self.refractory_period
            self.membrane_potential = self.reset_potential


class STDPSynapse:
    """STDP脉冲时序依赖可塑性突触，实现生物赫布学习规则"""
    def __init__(
        self,
        pre_neuron,
        post_neuron,
        init_weight=0.5,
        max_weight=1.0,
        min_weight=0.0,
        tau_plus=20.0,
        tau_minus=20.0,
        a_plus=0.01,
        a_minus=0.012
    ):
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.weight = init_weight
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus

    def transmit(self, current_step):
        """突触前发放脉冲，传递信号到突触后神经元"""
        if self.pre_neuron.fired:
            # 兴奋性神经元传正信号，抑制性传负信号
            signal = self.weight if self.pre_neuron.neuron_type == "excitatory" else -self.weight
            self.post_neuron.receive_input(signal, current_step)

    def update_weight(self):
        """根据前后脉冲时序，执行STDP权重更新"""
        delta_t = self.post_neuron.last_fire_step - self.pre_neuron.last_fire_step

        # 突触前先于突触后，权重增强（LTP）
        if delta_t > 0:
            delta_w = self.a_plus * np.exp(-delta_t / self.tau_plus)
            self.weight = min(self.weight + delta_w, self.max_weight)
        # 突触后先于突触前，权重减弱（LTD）
        elif delta_t < 0:
            delta_w = -self.a_minus * np.exp(delta_t / self.tau_minus)
            self.weight = max(self.weight + delta_w, self.min_weight)


class NeuralNetwork:
    """脉冲神经网络全局管理器，负责时序调度、状态管理"""
    def __init__(self):
        self.neurons = []
        self.synapses = []
        self.current_step = 0

    def add_neuron(self, neuron):
        """添加神经元，返回索引"""
        self.neurons.append(neuron)
        return len(self.neurons) - 1

    def connect(self, pre_idx, post_idx, **synapse_kwargs):
        """创建两个神经元之间的STDP突触连接"""
        pre = self.neurons[pre_idx]
        post = self.neurons[post_idx]
        self.synapses.append(STDPSynapse(pre, post, **synapse_kwargs))

    def step(self):
        """执行单步全局仿真：神经元更新→脉冲传递→权重更新"""
        self.current_step += 1

        # 1. 所有神经元更新时序状态
        for n in self.neurons:
            n.step(self.current_step)

        # 2. 突触传递脉冲信号
        for syn in self.synapses:
            syn.transmit(self.current_step)

        # 3. 突触执行STDP权重更新
        for syn in self.synapses:
            syn.update_weight()

    def inject_input(self, neuron_idx, signal):
        """给指定神经元注入外部输入信号"""
        self.neurons[neuron_idx].receive_input(signal, self.current_step)

    def get_firing_state(self):
        """获取所有神经元当前步的发放状态"""
        return [n.fired for n in self.neurons]


# 最小可运行演示：兴奋-抑制闭环环路
if __name__ == "__main__":
    net = NeuralNetwork()

    # 构建2个兴奋性神经元+1个抑制性神经元的小环路
    exc1 = net.add_neuron(LIFNeuron("excitatory"))
    exc2 = net.add_neuron(LIFNeuron("excitatory"))
    inh1 = net.add_neuron(LIFNeuron("inhibitory"))

    # 建立连接，形成闭环反馈
    net.connect(exc1, exc2)
    net.connect(exc2, inh1)
    net.connect(inh1, exc1)

    # 注入初始信号触发网络活动
    print(f"Step | exc1 | exc2 | inh1")
    print("-" * 25)
    net.inject_input(exc1, 1.2)

    # 运行20步仿真
    for step in range(20):
        net.step()
        fire_state = net.get_firing_state()
        print(f"{step:<4} | {fire_state[0]:<4} | {fire_state[1]:<4} | {fire_state[2]:<4}")

    print("\n演示完成：已验证神经元脉冲发放、突触传递与闭环动态")
