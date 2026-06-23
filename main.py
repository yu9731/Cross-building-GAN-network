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

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Dense, Concatenate

class MMoE(tf.keras.layers.Layer):
    def __init__(self, num_experts, num_tasks, expert_filters=256, kernel_size=1, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.expert_filters = expert_filters
        self.kernel_size = kernel_size
        # experts: Conv2D layers
        self.experts = [tf.keras.layers.Conv2D(expert_filters, kernel_size, padding='same', activation='relu')
                        for _ in range(num_experts)]
        # gates: per-task small Dense nets (we'll derive gate logits from global pooled features)
        self.gate_dense = [tf.keras.Sequential([
                                tf.keras.layers.GlobalAveragePooling2D(),
                                tf.keras.layers.Dense(num_experts)  # logits -> softmax
                             ]) for _ in range(num_tasks)]

    def call(self, x):  # x is a 4D tensor (batch, H, W, C)
        expert_outs = [expert(x) for expert in self.experts]  # length=num_experts
        expert_stack = tf.stack(expert_outs, axis=1)
        task_outputs = []
        for t in range(self.num_tasks):
            logits = self.gate_dense[t](x)
            gates = tf.nn.softmax(logits, axis=-1)
            gates_exp = tf.reshape(gates, (-1, self.num_experts, 1, 1, 1))
            mixed = tf.reduce_sum(expert_stack * gates_exp, axis=1)
            task_outputs.append(mixed)
        return task_outputs

for cnt in range(3):
    i = 0
    while i < len(building_combo_lst):
        x_train_lst = []
        building_combo = building_combo_lst[i]
        for building in building_combo:
            x_train_1 = np.load(f'/kaggle/input/when2heat-gf/task_{building}_data_solar_24.npy')
            x_train_lst.append(x_train_1)
            
        batch_size = 32
        # GF, hotel: 32, When2Heat: 32/64
        pred_hour = 24
        variable_num = 4
        num_tasks = building_num
    
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
                                     kernel_initializer=initializer, use_bias=False))  #kernel_regularizer=tf.keras.regularizers.l2(l2=1e-4),
        
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
            inputs = [tf.keras.layers.Input(shape=[pred_hour, variable_num, 1], name=f'input_task{t}') for t in range(num_tasks)]
            
            down_stacks = [
                downsample(base, 3, (2,2), apply_batchnorm=False),    # 64, 128
                downsample(base*2, 3, strides),                        # 128, 256
                dilated(base*2, 3, (1, 1)),  # 256 or 512
                dilated(base*2, 3, (2, 2)),
                dilated(base*2, 3, (4, 4)),
                dilated(base*2, 3, (8, 8)),
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
    
            shared_mmoe = MMoE(num_experts=num_tasks, num_tasks=num_tasks)
            experts_out = [shared_mmoe(b)[t] for t, b in enumerate(bottlenecks)]  # call MMoE per task bottleneck
    
            up_stacks = []
            decoder_vars_per_task = []
            x_decoded = list(experts_out)
            for t in range(num_tasks):
                up_stacks.append([
                    upsample(base*2, 3, (1,1)),
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
        
            return generator, decoder_vars_per_task
    
        def Discriminator(pred_hour, strides, variable_num):
            initializer = tf.random_normal_initializer(0., 0.02)
        
            inp = tf.keras.layers.Input(shape=[pred_hour, variable_num, 1], name='input_image')
            tar = tf.keras.layers.Input(shape=[pred_hour, variable_num, 1], name='target_image')
        
            x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)
        
            down1 = downsample(32, 3, strides, False)(x)  # 32, 64
            down2 = downsample(64, 3, strides)(down1)  # 64
            down3 = downsample(128, 3, strides)(down2)  # 128
        
            zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
            conv = tf.keras.layers.Conv2D(128, 3, strides=1,
                                          kernel_initializer=initializer,
                                          use_bias=False)(zero_pad1)  # 128
        
            batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
        
            leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
        
            zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)
        
            last = tf.keras.layers.Conv2D(1, 3, strides=1,
                                          kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)
        
            return tf.keras.Model(inputs=[inp, tar], outputs=last)
        
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
        
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
        def generator_loss(disc_generated_output, gen_output, target):
          gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
          l2_loss = tf.reduce_mean(tf.abs(target - gen_output))
          total_gen_loss = gan_loss + (100 * l2_loss)
          return total_gen_loss
        
        def discriminator_loss(disc_real_output, disc_generated_output):
          real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
          generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
          total_disc_loss = real_loss + generated_loss
          return total_disc_loss
    
        def soft_param_sharing_loss(decoder_vars_per_task, lambda_reg=0.01):
            reg_loss = 0.0
            for i in range(num_tasks):
                for j in range(i+1, num_tasks):
                    for v3, v4 in zip(decoder_vars_per_task[i], decoder_vars_per_task[j]):
                        reg_loss += tf.reduce_sum(tf.square(v3 - v4))
            return lambda_reg * reg_loss
    
        generator, decoder_vars_per_task = Generator(pred_hour, (2,2), 128)
        discriminators = [Discriminator(pred_hour, (2,2), variable_num) for _ in range(num_tasks)]
        
        opt_g = tf.keras.optimizers.Adam(1e-4)  #1e-4  When2Heat:5e-4/1e-4
        opt_d_list = [tf.keras.optimizers.Adam(1e-4) for _ in range(num_tasks)]  #1e-4  When2Heat:5e-4/1e-4
        
        def make_dataset(x):
            ds = tf.data.Dataset.from_tensor_slices(x)
            ds = ds.shuffle(buffer_size=len(x))
            ds = ds.take(int(len(x))).batch(batch_size).repeat()   #0.8 * 
            return ds.map(lambda batch: tf.cast(batch, tf.float32),
                          num_parallel_calls=tf.data.AUTOTUNE)
        
        datasets   = [make_dataset(x) for x in x_train_lst]
        combined_ds = tf.data.Dataset.zip(tuple(datasets))
        ds_iter     = iter(combined_ds)
        steps_per_epoch = int(len(x_train_lst[0]) / batch_size)        
    
        log_sigma = tf.Variable(tf.zeros([num_tasks], dtype=tf.float32), trainable=True, name="log_sigma")
        
        @tf.function
        def train_step(batch_tuple, batch_size, variable_num):
            total_g_loss = 0.0
            total_d_loss = 0.0
            g_losses = []
            d_losses = []
        
            # Prepare masked generator inputs (same mask used across tasks in your original code)
            masks = get_points(pred_hour, batch_size, variable_num)
            masks = tf.cast(masks, tf.float32)
        
            # gen_inputs: apply mask to each task's batch
            gen_inputs = [tf.cast(batch_tuple[t], tf.float32) * (1.0 - masks) for t in range(num_tasks)]
            targets = [batch_tuple[t] for t in range(num_tasks)]
            #initial_task_losses = None
            with tf.GradientTape(persistent=True) as tape:
                gen_outputs = generator(gen_inputs, training=True)
                for t in range(num_tasks):
                    gen_out = gen_outputs[t]
                    gen_in = gen_inputs[t]
                    target = targets[t]
                    disc = discriminators[t]
        
                    real_out = disc([gen_in, target], training=True)
                    fake_out = disc([gen_out, target], training=True)  # used in G loss
                    
                    g_loss = generator_loss(fake_out, gen_out, target) # + tf.add_n(generator.losses)
                    d_loss = discriminator_loss(real_out, fake_out)
                    
                    g_losses.append(g_loss)
                    d_losses.append(d_loss)
    
                    total_d_loss += d_loss
    
                # Update generator (single optimizer)
                per_task_total_tensor = tf.stack(g_losses)   # shape (num_tasks,)
                weights = tf.exp(-2.0 * log_sigma)                # 1/sigma^2
                weighted = 0.5 * weights * per_task_total_tensor
                total_g_loss = tf.reduce_sum(weighted) + tf.reduce_sum(log_sigma)
                total_g_loss += soft_param_sharing_loss(decoder_vars_per_task, lambda_reg=0.01)
                                        
            gen_vars = generator.trainable_variables + [log_sigma]
            grads_g = tape.gradient(total_g_loss, gen_vars)
            opt_g.apply_gradients(zip(grads_g, gen_vars))
            #grads_g, _ = tf.clip_by_global_norm(grads_g, 5.0)
            #opt_g.apply_gradients(zip(grads_g, gen_vars))
        
            # Update discriminators separately
            for t in range(num_tasks):
                grads_d = tape.gradient(d_losses[t], discriminators[t].trainable_variables)
                opt_d_list[t].apply_gradients(zip(grads_d, discriminators[t].trainable_variables))
    
            #del tape
        
            return total_g_loss, total_d_loss
    
        for epoch in range(201):
            for _ in range(steps_per_epoch):
                batch = next(ds_iter)
                bs = batch[0].shape[0]
                g_loss, d_loss = train_step(batch, bs, variable_num)
        
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d}: G_loss={g_loss:.4f}, D_loss={d_loss:.4f}")
                generator.save(f'pre_trained_model/MMoE_{location_lst[i]}_shared_encoder_bottleneck_generator.h5')
        i += 1
