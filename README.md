# Code of ATSRec (Adversarial Training for Representation Degeneration Problem in Sequential Recommendation)

Run following command to play with the code in scr:

For the base model GRU4Rec, NARM, SASRec and BERT4Rec without adversarial training, Such as SASRec in dataset Toys_and_Games:
```
python main.py --gpu_id=0 --data_name=Toys_and_Games --base_model_name=SASRec --epochs=300 --with_AT=No 
```
### result:
'HIT@5': '0.0479', 'HIT@10': '0.0698', 'HIT@20': '0.0970', 'NDCG@5': '0.0334', 'NDCG@10': '0.0405', 'NDCG@20': '0.0473'


For our ATSRec, the base model with adversarial training

Such as SASRec in dataset Toys_and_Games:
```
python main.py --gpu_id=0 --data_name=Toys_and_Games --base_model_name=SASRec --epochs=300 --with_AT=Yes --attack_train=pgd --adv_step=5 --epsilon=1.0 --eta=0.7
```
### result:
'HIT@5': '0.0545', 'HIT@10': '0.0777', 'HIT@20': '0.1086', 'NDCG@5': '0.0370', 'NDCG@10': '0.0444', 'NDCG@20': '0.0522'
