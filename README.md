# stable-diffusion

This project implements a stable diffusion model using a variational encoder with self-attention and a diffusion model with cross attention.

## Models Architecture
- ### Variational Autoencoder
    Consists of an encoder and a decoder, both utilizing self-attention. The encoder is used to convert samples into a lower latent representation. 
This representation can be passed to the diffusion model. The result of the diffusion model can then be decoded using the decoder.
- ### Diffusion Model
    Model is based on UNet architecture utilizing cross attention. 
Residual blocks also use time steps embeddings, which are created using sine/cosine positional encoding.