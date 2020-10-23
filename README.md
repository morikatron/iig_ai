# AI for Imperfect Information Game  
このリポジトリは不完全情報ゲームのナッシュ均衡戦略を計算的に求めるアルゴリズム、Counterfactual Regret Minimization(CFR)をPython3で実装したものです。
CFRについて解説したブログは[こちら](https://tech.morikatron.ai/entry/2020/08/31/100000 "こちら")になります  

## Relevant Papers
- An Introduction to Counterfactual Regret Minimization, T. Neller, M. Lanctot 2013
http://modelai.gettysburg.edu/2013/cfr/  

- Regret Minimization in Games with Incomplete Information, M. Zinkevich, M. Bowling, M. Johanson, C. Piccione. NIPS 2007.  
http://martin.zinkevich.org/publications/regretpoker.pdf  

## Requirements
 - Python3
 - pyyaml
 - tqdm

## Usage
  - clone this repo
 ```
 $ git clone https://github.com/morikatron/iig_ai.git
 ```
  - change directory and run 
 ```
 $ cd iig_ai
 $ python cfr/cfr.py
 ```
 
## Performance Example
 ![vanilla cfr performance](https://github.com/morikatron/iig_ai/blob/master/assets/vanilla_cfr_performance.png)
