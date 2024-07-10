## Self Supervised Representation Learning for Music

[ 1. Same embedding space]: <>
[ 2. Contrastive learning]: <>
[ 3. Negative Sampling]: <>
[ 4. Audio Features]: <> 
[ 5. Pass uncomputed song embeddings to fusion model]: <>


[ 6. Stream for more number of songs]: <>
[ 7. More fusion methods]: <>

To run for evaluation:
`python run.py --eval 1 --train 0`

### Project Description:

#### Overview

This project aims to develop a sophisticated music recommendation system that leverages both audio and textual (lyrical) data to understand and suggest music. The core idea is to encode and combine these two modalities to create rich and robust embeddings that capture the nuances of both sound and lyrics, leading to more accurate and personalized music recommendations.

#### Key Components

**Data Preparation:**

- **Audio Data:** Features are extracted and normalized from audio files.
- **Text Data:** Lyrics are embedded using pre-trained language models.
- **Dataset Creation:** The dataset is constructed with pairs of audio and text embeddings, along with corresponding labels and negative samples for training.

**Model Architecture:**

- **Encoder:**
  - **Audio Encoder:** Encodes audio features into a fixed-dimensional embedding.
  - **Text Encoder:** Encodes lyrics into a fixed-dimensional embedding.
  - **Cross-Attention Mechanism:** Allows the model to focus on relevant parts of the text and audio when creating embeddings.
  - **Self-Attention Mechanism:** Helps in capturing internal dependencies within the audio and text modalities.
- **Decoder:**
  - Maps the combined embeddings to a set of classes representing the songs in the dataset.
  - Uses a linear layer to produce logits for classification.

**Training Process:**

- **Loss Function:** Utilizes a custom `SkipGram_NegSample_Loss` which combines the negative sampling technique used in word2vec models with the binary cross-entropy loss for better training stability.
- **Optimization:** Uses Adam optimizer with a learning rate scheduler to adjust the learning rate based on validation performance.
- **Training Loop:** Involves training the model on batches of data, calculating the loss, performing backpropagation, and updating the model parameters.

**Evaluation:**

- **Evaluation Metrics:** Measures the cosine similarity between embeddings of the recommended and actual songs.
- **Relative Evaluation:** Computes similarity scores with the neighbors or randomly selected songs to gauge the performance.

**Logging and Checkpointing:**

- Utilizes `wandb` (Weights & Biases) for logging training metrics and visualizations.
- Periodically saves model checkpoints to facilitate resuming training and evaluating performance.

**Hyperparameters and Configuration:**

- Configurations such as the number of epochs, learning rate, embedding dimensions, and negative samples are specified via command-line arguments.
- Allows flexible experimentation with different settings to optimize the model's performance.

#### Objectives

- **Multimodal Embedding:** Effectively combine audio and textual data to create rich song embeddings.
- **Accurate Recommendations:** Improve the accuracy of music recommendations by capturing more nuanced relationships between songs.
- **Efficient Training:** Develop a robust training pipeline that includes effective loss functions, optimization techniques, and learning rate schedules.
- **Comprehensive Evaluation:** Implement thorough evaluation mechanisms to ensure the model performs well in real-world scenarios.

#### Conclusion

This project represents a comprehensive approach to building a multimodal music recommendation system. By leveraging both audio and textual data, and employing advanced neural network techniques such as cross-attention and self-attention mechanisms, the system aims to provide more personalized and accurate music recommendations.
