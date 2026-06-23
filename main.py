import numpy as np
import tensorflow as tf
import datetime

if (tf.test.is_gpu_available):
    print("GPU")
else:
    print("CPU")

# List all physical GPUs available to TensorFlow
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)

from tensorflow.python.client import device_lib

device_lib.list_local_devices()

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPU.")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print("Error enabling memory growth:", e)

  from itertools import combinations

building_combo_lst = [['AT_SFH', 'AT_COM', 'AT_MFH']]
location_lst = ['AT']

building_num = 3
pred_hour = 24

import os, numpy as np, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

@tf.keras.utils.register_keras_serializable()
class MMoE(tf.keras.layers.Layer):
    def __init__(self, num_experts, num_tasks, expert_filters=256, kernel_size=1,
                 gate_hidden=128, temperature=0.5,
                 entropy_weight=1e-3, diversity_weight=1e-2, balance_weight=1e-3,
                 init_bias_strength=1.5, **kwargs):
        super().__init__(**kwargs)
        self.num_experts, self.num_tasks = num_experts, num_tasks
        self.expert_filters, self.kernel_size = expert_filters, kernel_size
        self.gate_hidden, self.temperature = gate_hidden, temperature
        self.entropy_weight, self.diversity_weight, self.balance_weight = entropy_weight, diversity_weight, balance_weight
        self.init_bias_strength = init_bias_strength
        self.experts = [tf.keras.Sequential([
            tf.keras.layers.Conv2D(expert_filters, kernel_size, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(expert_filters, 1, padding="same", activation="relu")
        ], name=f"expert_{e+1}") for e in range(num_experts)]
        self.gate_nets = [tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(gate_hidden, activation="relu"),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(num_experts)
        ], name=f"gate_task_{t+1}") for t in range(num_tasks)]
    def build(self, input_shape):
        init = np.zeros((self.num_tasks, self.num_experts), dtype=np.float32)
        for t in range(self.num_tasks):
            init[t, t % self.num_experts] = self.init_bias_strength
        self.task_expert_bias = self.add_weight(name="task_expert_bias", shape=(self.num_tasks, self.num_experts), initializer=tf.keras.initializers.Constant(init), trainable=True)
        super().build(input_shape)
    def call(self, x_list, return_gates=False, training=None):
        if not isinstance(x_list, (list, tuple)):
            x_list = [x_list for _ in range(self.num_tasks)]
        task_outputs, gate_outputs = [], []
        for t in range(self.num_tasks):
            x = x_list[t]
            expert_outs = [expert(x, training=training) for expert in self.experts]
            expert_stack = tf.stack(expert_outs, axis=1)
            logits = self.gate_nets[t](x, training=training) + self.task_expert_bias[t]
            gates = tf.nn.softmax(logits / self.temperature, axis=-1)
            gates_exp = tf.reshape(gates, (-1, self.num_experts, 1, 1, 1))
            mixed = tf.reduce_sum(expert_stack * gates_exp, axis=1)
            task_outputs.append(mixed)
            gate_outputs.append(gates)
        gates_stack = tf.stack(gate_outputs, axis=1)
        entropy = -tf.reduce_mean(tf.reduce_sum(gates_stack * tf.math.log(gates_stack + 1e-8), axis=-1))
        self.add_loss(self.entropy_weight * entropy)
        task_mean = tf.reduce_mean(gates_stack, axis=0)
        task_norm = tf.math.l2_normalize(task_mean, axis=-1)
        sim = tf.matmul(task_norm, task_norm, transpose_b=True)
        if self.num_tasks > 1:
            off_diag_sim = (tf.reduce_sum(sim) - tf.cast(self.num_tasks, tf.float32)) / tf.cast(self.num_tasks * (self.num_tasks - 1), tf.float32)
            self.add_loss(self.diversity_weight * off_diag_sim)
        expert_usage = tf.reduce_mean(gates_stack, axis=[0, 1])
        balance_loss = tf.reduce_sum(tf.square(expert_usage - 1.0 / self.num_experts))
        self.add_loss(self.balance_weight * balance_loss)
        return (task_outputs, gate_outputs) if return_gates else task_outputs
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_experts": self.num_experts, "num_tasks": self.num_tasks,
            "expert_filters": self.expert_filters, "kernel_size": self.kernel_size,
            "gate_hidden": self.gate_hidden, "temperature": self.temperature,
            "entropy_weight": self.entropy_weight, "diversity_weight": self.diversity_weight,
            "balance_weight": self.balance_weight, "init_bias_strength": self.init_bias_strength
        })
        return config

def upsample(filters, size, strides):
    init = tf.random_normal_initializer(0., 0.02)
    return tf.keras.Sequential([tf.keras.layers.Conv2DTranspose(filters, size, strides=strides, padding="same", kernel_initializer=init, use_bias=False), tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU()])

def downsample(filters, size, strides, apply_batchnorm=True):
    init = tf.random_normal_initializer(0., 0.02)
    layers = [tf.keras.layers.Conv2D(filters, size, strides=strides, padding="same", kernel_initializer=init, use_bias=False)]
    if apply_batchnorm: layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.LeakyReLU())
    return tf.keras.Sequential(layers)

def dilated(filters, size, dilation_rate):
    init = tf.random_normal_initializer(0., 0.02)
    return tf.keras.Sequential([tf.keras.layers.Conv2D(filters, size, strides=1, padding="same", dilation_rate=dilation_rate, kernel_initializer=init, use_bias=False), tf.keras.layers.ReLU()])

def Generator(pred_hour, strides, base, variable_num, num_tasks):
    inputs = [tf.keras.layers.Input(shape=[pred_hour, variable_num, 1], name=f"input_task{t+1}") for t in range(num_tasks)]
    down_stacks = [downsample(base, 3, (2,1), False), downsample(base*2, 3, strides), dilated(256, 3, (1,1)), dilated(256, 3, (2,2)), dilated(256, 3, (4,4)), dilated(256, 3, (8,8))]
    x_list, skips_list = list(inputs), [[] for _ in range(num_tasks)]
    for layer in down_stacks:
        for t in range(num_tasks):
            x_list[t] = layer(x_list[t])
            skips_list[t].append(x_list[t])
    bottlenecks = [skips_list[t][-1] for t in range(num_tasks)]
    skips_for_decoder = [skips_list[t][0:2][::-1] for t in range(num_tasks)]
    shared_mmoe = MMoE(num_experts=num_tasks, num_tasks=num_tasks, expert_filters=base*2,
    gate_hidden=128, temperature=0.5, entropy_weight=1e-3, diversity_weight=1e-2,
    balance_weight=1e-3, init_bias_strength=1.5,name="shared_mmoe")

    experts_out, gate_outputs = shared_mmoe(bottlenecks, return_gates=True)
    up_stacks, decoder_vars_per_task, x_decoded = [], [], list(experts_out)
    for t in range(num_tasks):
        up_stacks.append([upsample(base*2, 3, (1,1)), upsample(base, 3, strides)])
        decoder_vars_per_task.append([])
    for layer_idx in range(len(up_stacks[0])):
        for t in range(num_tasks):
            x_decoded[t] = up_stacks[t][layer_idx](x_decoded[t])
            x_decoded[t] = tf.keras.layers.Concatenate()([x_decoded[t], skips_for_decoder[t][layer_idx]])
            decoder_vars_per_task[t].extend(up_stacks[t][layer_idx].trainable_variables)
    outputs, init = [], tf.random_normal_initializer(0., 0.02)
    for t in range(num_tasks):
        last = tf.keras.layers.Conv2DTranspose(1, 3, strides=(2,1), padding="same", kernel_initializer=init, activation="softplus", name=f"output_task{t+1}")
        x_decoded[t] = last(x_decoded[t])
        decoder_vars_per_task[t].extend(last.trainable_variables)
        outputs.append(x_decoded[t])
    generator = tf.keras.Model(inputs=inputs, outputs=outputs, name="Generator_MMoE")
    gate_model = tf.keras.Model(inputs=inputs, outputs=gate_outputs, name="MMoE_gate_model")
    return generator, gate_model, decoder_vars_per_task

def Discriminator(pred_hour, strides, variable_num):
    init = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[pred_hour, variable_num, 1], name="input_image")
    tar = tf.keras.layers.Input(shape=[pred_hour, variable_num, 1], name="target_image")
    x = tf.keras.layers.concatenate([inp, tar])
    x = downsample(32, 3, strides, False)(x); x = downsample(64, 3, strides)(x); x = downsample(128, 3, strides)(x)
    x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=1, kernel_initializer=init, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.LeakyReLU()(x); x = tf.keras.layers.ZeroPadding2D()(x)
    out = tf.keras.layers.Conv2D(1, 3, strides=1, kernel_initializer=init)(x)
    return tf.keras.Model(inputs=[inp, tar], outputs=out)

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    return gan_loss + 100.0 * l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    return loss_object(tf.ones_like(disc_real_output), disc_real_output) + loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

def soft_param_sharing_loss(decoder_vars_per_task, num_tasks, lambda_reg=0.01):
    reg_loss = 0.0
    for i in range(num_tasks):
        for j in range(i+1, num_tasks):
            for v_i, v_j in zip(decoder_vars_per_task[i], decoder_vars_per_task[j]):
                reg_loss += tf.reduce_sum(tf.square(v_i - v_j))
    return lambda_reg * reg_loss

def make_dataset(x, batch_size):
    return tf.data.Dataset.from_tensor_slices(x).shuffle(len(x)).batch(batch_size, drop_remainder=True).repeat().map(lambda b: tf.cast(b, tf.float32), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

i = 0
while i < len(building_combo_lst):
    x_train_lst, building_combo = [], building_combo_lst[i]
    for building in building_combo:
        x_train_lst.append(np.load(f"/kaggle/input/when2heat-gf/task_{building}_data_solar_24.npy"))
        # x_train_lst.append(np.load(f"/kaggle/input/hotel/task_{building}_data_24_TotalCons_min_max.npy"))
    batch_size, pred_hour, variable_num, load_idx = 32, 24, 4, -2
    num_tasks = building_num
    generator, gate_model, decoder_vars_per_task = Generator(pred_hour, (2,1), 128, variable_num, num_tasks)
    discriminators = [Discriminator(pred_hour, (2,1), variable_num) for _ in range(num_tasks)]
    opt_g = tf.keras.optimizers.Adam(1e-4)
    opt_d_list = [tf.keras.optimizers.Adam(1e-4) for _ in range(num_tasks)]
    datasets = [make_dataset(x, batch_size) for x in x_train_lst]
    ds_iter = iter(tf.data.Dataset.zip(tuple(datasets)))
    steps_per_epoch = int(len(x_train_lst[0]) / batch_size)
    log_sigma = tf.Variable(tf.zeros([num_tasks], dtype=tf.float32), trainable=True, name="log_sigma")
    keep_mask_np = np.ones((1, 1, variable_num, 1), dtype=np.float32)
    keep_mask_np[:, :, load_idx, :] = 0.0
    load_keep_mask = tf.constant(keep_mask_np, dtype=tf.float32)

    @tf.function
    def train_step(batch_tuple):
        total_d_loss, g_losses, d_losses = 0.0, [], []
        gen_inputs = [tf.cast(batch_tuple[t], tf.float32) * load_keep_mask for t in range(num_tasks)]
        targets = [tf.cast(batch_tuple[t], tf.float32) for t in range(num_tasks)]
        with tf.GradientTape(persistent=True) as tape:
            gen_outputs = generator(gen_inputs, training=True)
            for t in range(num_tasks):
                gen_out, gen_in, target, disc = gen_outputs[t], gen_inputs[t], targets[t], discriminators[t]
                real_out = disc([gen_in, target], training=True)
                fake_out = disc([gen_in, gen_out], training=True)
                g_loss = generator_loss(fake_out, gen_out, target)
                d_loss = discriminator_loss(real_out, fake_out)
                g_losses.append(g_loss); d_losses.append(d_loss); total_d_loss += d_loss
            per_task_loss = tf.stack(g_losses)
            weights = tf.exp(-2.0 * log_sigma)
            total_g_loss = tf.reduce_sum(0.5 * weights * per_task_loss) + tf.reduce_sum(log_sigma)
            total_g_loss += soft_param_sharing_loss(decoder_vars_per_task, num_tasks, lambda_reg=0.01)
        gen_vars = generator.trainable_variables + [log_sigma]
        opt_g.apply_gradients(zip(tape.gradient(total_g_loss, gen_vars), gen_vars))
        for t in range(num_tasks):
            opt_d_list[t].apply_gradients(zip(tape.gradient(d_losses[t], discriminators[t].trainable_variables), discriminators[t].trainable_variables))
        del tape
        return total_g_loss, total_d_loss

    for epoch in range(201):
        for _ in range(steps_per_epoch):
            batch = next(ds_iter)
            g_loss, d_loss = train_step(batch)
        if epoch % 10 == 0:
            print(f"Run={cnt}, Combo={i}, Epoch={epoch:03d}: G_loss={g_loss:.4f}, D_loss={d_loss:.4f}")
            os.makedirs("pre_trained_model", exist_ok=True)
            generator.save(f"pre_trained_model/MMoE_{location_lst[i]}_generator.keras")
            gate_model.save(f"pre_trained_model/MMoE_{location_lst[i]}_gate_model.keras")
        
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d}: G_loss={g_loss:.4f}, D_loss={d_loss:.4f}")
                generator.save(f'pre_trained_model/MMoE_{location_lst[i]}_shared_encoder_bottleneck_generator.h5')
        i += 1
