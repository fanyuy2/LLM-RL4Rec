# LLM-RL4Rec

Tasks By Friday Oct 25, 2024

Rishabh & Rutik 
- report 
- EDA

Yash 
- prompt
- LLM pipeline 
- Evaluation

Fanyu
- CF pipeline 
- Evaluation

- PPO - https://drive.google.com/file/d/1FGN1jHT1uOyKCJfoTbjpjLzQc934cFgF/view?usp=sharing

| **Stage**             | **MAP**      | **NDCG@k**          | **Precision@k**  | **Recall@k**       |
|------------------------|--------------|---------------------|------------------|--------------------|
| **Pre-SFT (on 100k)**           | 0.0061       | 0.0062              | 0.0061           | 0.0051             |
| **1M Pre-SFT**        | 0.0025       | 0.0021              | 0.0025           | 0.0023             |
| **Post-SFT (on 100k)**          | 0.0182       | 0.0169              | 0.0182           | 0.0144             |
| **1M Post-SFT**       | NaN          | 0.0234              | 0.0220           | 0.0181             |
| **RLHF (PPO on 100k)**| 0.0182       | 0.0167              | 0.0182           | 0.0156             |

- Pre SFT
 {'MAP': 0.006060606060606061,
 'NDCG@k': 0.006245369396893475,
 'Precision@k': 0.006060606060606061,
 'Recall@k': 0.0051226551226551224}

 - 1M
   {'MAP': np.float64(0.002546296296296296),
 'NDCG@k': np.float64(0.002130172178852872),
 'Precision@k': 0.0025462962962962965,
 'Recall@k': 0.0022872574955908292}

- Post SFT
{'MAP': 0.01818181818181818,
 'NDCG@k': 0.016869970668818752,
 'Precision@k': 0.01818181818181818,
 'Recall@k': 0.01441197691197691}

 - 1M
   {'MAP': np.float64(nan),
 'NDCG@k': np.float64(0.0234145817875562),
 'Precision@k': 0.02199074074074074,
 'Recall@k': 0.018121693121693122}

 - RLHF (PPO on 100k)
{'MAP': np.float64(0.01818181818181818), 'NDCG@k': np.float64(0.0166918513294695), 'Precision@k': 0.01818181818181818, 'Recall@k': 0.015593434343434341}
