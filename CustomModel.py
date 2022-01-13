import tensorflow as tf
from tensorflow import keras

class CustomModel(keras.Model):
    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, w = data
        else:
            x, y = data
            
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True) # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            if len(data) == 3:
                loss = self.compiled_loss(
                    y,
                    y_pred,
                    sample_weight=w)
                
            else:
                loss = self.compiled_loss(
                    y,
                    y_pred)
        
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
