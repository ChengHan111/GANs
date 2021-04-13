In order to ***load Pretrained Model***
We can have a larger batch size since we are loading the pretrained model, in train.py val_loader, you can change the batch_size from 1 to 8

Comment out 
```python
train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,)
    if config.SAVE_MODEL and epoch % 5 == 0:
        save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
        save_checkpoint(gen, opt_disc, filename=config.CHECKPOINT_DISC)
```
       
in train.py

Comment out  
```python
optimizer.load_state_dict(checkpoint["optimizer"])
```

in utils.py

Turn 
```python 
LOAD_MODEL = True
``` 
in config.py