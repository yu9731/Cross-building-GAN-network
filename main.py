import os, numpy as np, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

if (tf.test.is_gpu_available):
    print("GPU")
else:
    print("CPU")

# List all physical GPUs available to TensorFlow
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)

building_combo_lst = [['AT_SFH', 'CH_SFH', 'DE_SFH'], ['AT_COM', 'CH_COM', 'DE_COM'], ['AT_MFH', 'CH_MFH', 'DE_MFH']]
# [['AT_SFH', 'AT_COM', 'AT_MFH'], ['CH_SFH', 'CH_COM', 'CH_MFH'], ['DE_SFH', 'DE_COM', 'DE_MFH']]
location_lst = ['SFH', 'COM', 'MFH']
# ['AT', 'CH', 'DE']

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
        ], name=f"expert_{e + 1}") for e in range(num_experts)]
        self.gate_nets = [tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(gate_hidden, activation="relu"),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(num_experts)
        ], name=f"gate_task_{t + 1}") for t in range(num_tasks)]

    def build(self, input_shape):
        init = np.zeros((self.num_tasks, self.num_experts), dtype=np.float32)
        for t in range(self.num_tasks):
            init[t, t % self.num_experts] = self.init_bias_strength
        self.task_expert_bias = self.add_weight(name="task_expert_bias", shape=(self.num_tasks, self.num_experts),
                                                initializer=tf.keras.initializers.Constant(init), trainable=True)
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
        all_gates = tf.concat(gate_outputs, axis=0)   #, axis=1)
        entropy = -tf.reduce_mean(tf.reduce_sum(all_gates * tf.math.log(all_gates + 1e-8), axis=-1))
        self.add_loss(self.entropy_weight * entropy)
        task_mean = tf.stack([tf.reduce_mean(g, axis=0) for g in gate_outputs], axis=0)
        task_norm = tf.math.l2_normalize(task_mean, axis=-1)
        sim = tf.matmul(task_norm, task_norm, transpose_b=True)
        if self.num_tasks > 1:
            off_diag_sim = (tf.reduce_sum(sim) - tf.cast(self.num_tasks, sim.dtype)) / tf.cast(
                self.num_tasks * (self.num_tasks - 1), sim.dtype)
            self.add_loss(self.diversity_weight * off_diag_sim)
        expert_usage = tf.reduce_mean(all_gates, axis=0)
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

import numpy as np

sample_num_combo = [[730, 730, 730]]
x_train_lst = [[] for k in range(len(building_combo_lst))]
batch_size_lst = [[] for k in range(len(building_combo_lst))]
building_num = 3
i = 0
while i < len(building_combo_lst):
    building_combo = building_combo_lst[i]
    for k, sample_num_lst in enumerate(sample_num_combo):
        x_train_ = []
        batch_size_ = []
        for l, sample_num in enumerate(sample_num_lst):
            if l < 1:
                x_train_1 = np.load(f'data/task_{building_combo[l]}_data_solar_24.npy')
                x_train_.append(x_train_1)
                batch_size_.append(32)
            else:
                x_train_1 = np.load(f'MTL/data/when2heat-gf/task_{building_combo[l]}_data_solar_24.npy')  # [150:150+sample_num//2]

                #x_train_1_cold = np.load(f'MTL/data/hotel/task_{building_combo[l]}_data_24_TotalCons_min_max.npy')#[0:sample_num // 2]
                #x_train_1_warm = np.load(f'MTL/data/hotel/task_{building_combo[l]}_data_24_TotalCons_min_max.npy')#[90:90+sample_num//2]
                #x_train_1 = np.vstack((x_train_1_cold, x_train_1_warm))

                x_train_.append(x_train_1)
                batch_size_.append(int(32 / (730 / sample_num)) if int(32 / (730 / sample_num)) > 1 else 1)
        x_train_lst[i].append(x_train_)
        batch_size_lst[i].append(batch_size_)
    i += 1

j = 0
while j < len(building_combo_lst):
    building_combo = building_combo_lst[j]
    x_train = x_train_lst[j]
    batch_size = batch_size_lst[j]

    for k in range(len(x_train)):
        Batch_size_lst = batch_size[k]
        X_train = x_train[k]
        sample_num_lst = sample_num_combo[k]

        pred_hour = 24
        variable_num = 4
        strides = (2, 2)
        base = 128
        num_tasks = 3

    def upsample(filters, size, strides):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=strides,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            # kernel_regularizer=tf.keras.regularizers.l2(l2=1e-4),
                                            use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.ReLU())
        return result


    def downsample(filters, size, strides, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',
                                   kernel_initializer=initializer,
                                   use_bias=False))  # kernel_regularizer=tf.keras.regularizers.l2(l2=1e-4),

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())
        return result


    def dilated(filters, size, dilation_rate):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                                          dilation_rate=dilation_rate,
                                          kernel_initializer=initializer,
                                          use_bias=False))
        result.add(tf.keras.layers.ReLU())

        return result


    def Generator(pred_hour, strides, base):
        inputs = [tf.keras.layers.Input(shape=[pred_hour, variable_num, 1], name=f'input_task{t}') for t in
                  range(num_tasks)]

        down_stacks = [
            downsample(base, 3, (2, 2), apply_batchnorm=False),
            downsample(base * 2, 3, strides),
            dilated(base * 2, 3, (1, 1)),
            dilated(base * 2, 3, (2, 2)),
            dilated(base * 2, 3, (4, 4)),
            dilated(base * 2, 3, (8, 8)),
        ]

        x_list = list(inputs)
        skips_list = [[] for _ in range(num_tasks)]
        for layer_idx in range(len(down_stacks)):
            for t in range(num_tasks):
                layer = down_stacks[layer_idx]
                x_list[t] = layer(x_list[t])
                skips_list[t].append(x_list[t])

        bottlenecks = [skips_list[t][-1] for t in range(num_tasks)]
        skips_for_decoder = [skips_list[t][0:2][::-1] for t in range(num_tasks)]  # matches your original 0:2

        encoder_model = tf.keras.Model(inputs=inputs, outputs=bottlenecks, name="Shared_encoder_model")

        shared_mmoe = MMoE(num_experts=num_tasks, num_tasks=num_tasks, expert_filters=base * 2,
                           gate_hidden=128, temperature=0.5, entropy_weight=1e-3, diversity_weight=1e-2,
                           balance_weight=1e-3, init_bias_strength=1.5, name="shared_mmoe")

        experts_out, gate_outputs = shared_mmoe(bottlenecks, return_gates=True)

        up_stacks = []
        decoder_vars_per_task = []
        x_decoded = list(experts_out)
        # x_decoded = list(bottlenecks)
        for t in range(num_tasks):
            up_stacks.append([
                upsample(base * 2, 3, (1, 1)),
                upsample(base, 3, strides),
            ])
            # Track decoder variables
            decoder_vars_per_task.append([])

        n_up = len(up_stacks[0])
        for layer_idx in range(n_up):
            for t in range(num_tasks):
                x_decoded[t] = up_stacks[t][layer_idx](x_decoded[t])
                skip = skips_for_decoder[t][layer_idx]
                x_decoded[t] = tf.keras.layers.Concatenate()([x_decoded[t], skip])
                decoder_vars_per_task[t].extend(up_stacks[t][layer_idx].trainable_variables)

        outputs = []
        initializer = tf.random_normal_initializer(0., 0.02)
        for t in range(num_tasks):
            last = tf.keras.layers.Conv2DTranspose(1, 3, strides=(2, 2), padding='same',
                                                   kernel_initializer=initializer,
                                                   activation='softplus')
            x_decoded[t] = last(x_decoded[t])
            decoder_vars_per_task[t].extend(last.trainable_variables)
            outputs.append(x_decoded[t])

        generator = tf.keras.Model(inputs=inputs, outputs=outputs, name="Generator_MMoE")
        gate_model = tf.keras.Model(inputs=inputs, outputs=gate_outputs, name="MMoE_gate_model")
        
        return generator, gate_model, encoder_model, decoder_vars_per_task


    def Discriminator(pred_hour, strides, variable_num):
        init = tf.random_normal_initializer(0., 0.02)
        inp = tf.keras.layers.Input(shape=[pred_hour, variable_num, 1], name="input_image")
        tar = tf.keras.layers.Input(shape=[pred_hour, variable_num, 1], name="target_image")
        x = tf.keras.layers.concatenate([inp, tar])
        x = downsample(32, 3, strides, False)(x)
        x = downsample(64, 3, strides)(x)
        x = downsample(128, 3, strides)(x)
        x = tf.keras.layers.ZeroPadding2D()(x)
        x = tf.keras.layers.Conv2D(128, 3, strides=1, kernel_initializer=init, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.ZeroPadding2D()(x)
        out = tf.keras.layers.Conv2D(1, 3, strides=1, kernel_initializer=init)(x)
        return tf.keras.Model(inputs=[inp, tar], outputs=out)

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def get_points(pred_hour, batch_size, variable_num):
        mask = []
        points = []
        for i in range(batch_size):
            m = np.zeros((pred_hour, variable_num, 1), dtype=np.uint8)
            x1 = np.random.randint(0, pred_hour - pred_hour + 1, 1)[0]
            x2 = x1 + pred_hour
            points.append([x1, x2])

            m[:, -2] = 1
            mask.append(m)
        return np.array(mask)

    def generator_loss(disc_generated_output, gen_output, target):
        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        return gan_loss + 100.0 * l1_loss

    def discriminator_loss(disc_real_output, disc_generated_output):
        return loss_object(tf.ones_like(disc_real_output), disc_real_output) + loss_object(
            tf.zeros_like(disc_generated_output), disc_generated_output)

    def soft_param_sharing_loss(decoder_vars_per_task, num_tasks, lambda_reg=0.01):
        reg_loss = 0.0
        for i in range(num_tasks):
            for j in range(i + 1, num_tasks):
                for v_i, v_j in zip(decoder_vars_per_task[i], decoder_vars_per_task[j]):
                    reg_loss += tf.reduce_sum(tf.square(v_i - v_j))
        return lambda_reg * reg_loss

    def make_dataset(x, batch_size):
        ds = tf.data.Dataset.from_tensor_slices(x)
        ds = ds.shuffle(buffer_size=len(x))
        ds = ds.take(int(len(x))).batch(batch_size).repeat()  # 0.8 *
        return ds.map(lambda batch: tf.cast(batch, tf.float32),
                      num_parallel_calls=tf.data.AUTOTUNE)

    datasets = [make_dataset(x, Batch_size_lst[j]) for j, x in enumerate(X_train)]
    combined_ds = tf.data.Dataset.zip(tuple(datasets))
    ds_iter = iter(combined_ds)
    steps_per_epoch = int(len(X_train[0]) / Batch_size_lst[0])

    generator, gate_model, encoder_model, decoder_vars_per_task = Generator(pred_hour, (2, 2), 128)
    discriminators = [Discriminator(pred_hour, (2, 2), variable_num) for _ in range(num_tasks)]

    opt_g = tf.keras.optimizers.Adam(1e-4)
    opt_d_list = [tf.keras.optimizers.Adam(1e-4) for _ in range(num_tasks)]

    log_sigma = tf.Variable(tf.zeros([num_tasks], dtype=tf.float32), trainable=True, name="log_sigma")
    generator.summary()

    building_suffix = ''
    for k in range(building_num):
        building_suffix += ('_' + str(building_combo[k]))

    @tf.function
    def train_step(batch_tuple, batch_size_lst, variable_num):
        total_d_loss = 0.0
        g_losses = []
        d_losses = []

        # Prepare masked generator inputs (same mask used across tasks in your original code)
        masks_lst = []
        for i, batch_size in enumerate(batch_size_lst):
            masks = get_points(pred_hour, batch_size, variable_num)
            masks = tf.cast(masks, tf.float32)
            masks_lst.append(masks)

        gen_inputs = [tf.cast(batch_tuple[t], tf.float32) * (1.0 - masks) for t, masks in enumerate(masks_lst)]
        targets = [batch_tuple[t] for t in range(num_tasks)]

        with tf.GradientTape(persistent=True) as tape:
            gen_outputs = generator(gen_inputs, training=True)
            for t in range(num_tasks):
                gen_out = gen_outputs[t]
                gen_in = gen_inputs[t]
                target = targets[t]
                disc = discriminators[t]

                real_out = disc([gen_in, target], training=True)
                fake_out = disc([gen_out, target], training=True)

                g_loss = generator_loss(fake_out, gen_out, target)
                d_loss = discriminator_loss(real_out, fake_out)

                g_losses.append(g_loss)
                d_losses.append(d_loss)

                total_d_loss += d_loss

            per_task_total_tensor = tf.stack(g_losses)  # shape (num_tasks,)
            weights = tf.exp(-2.0 * log_sigma)  # 1/sigma^2
            weighted = 0.5 * weights * per_task_total_tensor
            total_g_loss = tf.reduce_sum(weighted) + tf.reduce_sum(log_sigma)
            total_g_loss += soft_param_sharing_loss(decoder_vars_per_task, num_tasks, lambda_reg=0.01)

        gen_vars = generator.trainable_variables + [log_sigma]
        grads_g = tape.gradient(total_g_loss, gen_vars)
        opt_g.apply_gradients(zip(grads_g, gen_vars))

        for t in range(num_tasks):
            grads_d = tape.gradient(d_losses[t], discriminators[t].trainable_variables)
            opt_d_list[t].apply_gradients(zip(grads_d, discriminators[t].trainable_variables))

        return total_g_loss, total_d_loss

    for epoch in range(201):
        for _ in range(steps_per_epoch):
            batch = next(ds_iter)
            bs_lst = [batch[0].shape[0], batch[1].shape[0], batch[2].shape[0]]
            g_loss, d_loss = train_step(batch, bs_lst, variable_num)
        if epoch % 10 == 0:
            print(f"Epoch={epoch:03d}: G_loss={g_loss:.4f}, D_loss={d_loss:.4f}")
            generator.save(f"MTL/model/MMoE_{location_lst[j]}generator.keras")
            gate_model.save(f"pre_trained_model/MMoE_{location_lst[i]}_gate_model.keras")
            encoder_model.save(f"pre_trained_model/MMoE_{location_lst[i]}_encoder_model.keras")
    j += 1
