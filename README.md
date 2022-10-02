# LBox Open

A multi-task benchmark for Korean legal language understanding and judgement prediction by [LBox](https://lbox.kr)

# Authors

- [Wonseok Hwang](mailto:wonseok.hwang@lbox.kr)
- [Dongjun Lee](mailto:dongjun.lee@lbox.kr)
- [Kyoungyeon Cho](mailto:kycho@lbox.kr)
- [Hanuhl Lee](mailto:leehanuhl@lbox.kr)
- [Minjoon Seo](mailto:minjoon@lbox.kr)

# Updates
- Oct 2, 2022: [`defamation corpus-v0.1`](https://lbox-open.s3.ap-northeast-2.amazonaws.com/precedent_benchmark_dataset/defamation_corpus/defamation_corpus.jsonl) has been added. The corpus consists of 768 criminal cases related to "defamation (명예훼손)". The corpus will be integrated into `precedent corpus` in the future (at the moment, there can be some overlap between `precedent corpus` and `defamation corpus-v0.1`). See also [this issue](https://github.com/lbox-kr/lbox-open/issues/4#issue-1393652876).
- Sep 2022: Our paper is accepted for publication in NeurIPS 2022 Datasets and Benchmarks track! There will be major updates on the paper, the dataets, and the models soon! Meanwile, one can check the most recent version of our paper from [OpenReview](https://openreview.net/forum?id=TaARsI_Iio)
- Jun 2022: We release `lbox-open-v0.2`!
  - Two legal judgement prediction tasks, `ljp_criminal`, `ljp-civil`, are added to LBox Open.
  - `LCube-base`, a LBox Legal Language model with 124M parameters, is added.
  - The baseline scores and its training/test scripts are added.
  - Other updates
    - Some missing values in `facts` fields of `casename_classification` and `statute_classification` are updated.
    - `case_corpus` is renamed to `precedent_corpus`
- Mar 2022: We release `lbox-open-v0.1`!

# Paper

[A Multi-Task Benchmark for Korean Legal Language Understanding and Judgement Prediction](https://arxiv.org/abs/2206.05224)

# Benchmarks

- Last updated at Jun 16 2022
 
| **Model**         | casename       | statute        | ljp-criminal  | ljp-civil      | summarization    |
|-------------------|----------------|----------------|-----------------------------------------------------------------------|----------------|------------------|
|                   | EM             | EM             | F1-fine <br/>F1-imprisonment w/ labor<br/>F1-imprisonment w/o labor                                                                    | EM             | R1<br/>R2<br/>RL         | 
| KoGPT2            | $77.5 \pm 0.1$ | $85.7 \pm 0.8$ | $49.9 \pm 1.7$ <br/> $67.5 \pm 1.1$ <br/>  $69.2 \pm 1.6$                     | $64.6 \pm 2.0$ | $35.1$<br/> $24.2$<br/> $34.6$ | 
| LCube-base (ours) | $80.0 \pm 1.2$ | $87.6 \pm 0.5$ | $46.4 \pm 2.8$ <br/>   $69.3 \pm 0.3$<br/>   $70.3 \pm 0.7$                     | $68.0 \pm 0.6$ | $31.0$<br/> $20.7$<br/> $30.8$ | 
   - The errors are estimated from three independent experiments performed with different random seeds.
# Dataset

## How to use the dataset

We use [`datasets`](https://github.com/huggingface/datasets) library from `HuggingFace`.

```python
# !pip install datasets
from datasets import load_dataset

# casename classficiation task
data_cn = load_dataset("lbox/lbox_open", "casename_classification")

# statutes classification task
data_st = load_dataset("lbox/lbox_open", "statute_classification")

# Legal judgement prediction tasks
data_ljp_criminal = load_dataset("lbox/lbox_open", "ljp_criminal")
data_ljp_civil = load_dataset("lbox/lbox_open", "ljp_civil")

# case summarization task
data_summ = load_dataset("lbox/lbox_open", "summarization")

# precedent corpus
data_corpus = load_dataset("lbox/lbox_open", "precedent_corpus")


```

- [Explore the dataset on Colab](https://colab.research.google.com/drive/1R4T91Ix__-4rjtxATh7JeTX69zYrmWy0?usp=sharing)

## Dataset Description
### `precedent_corpus`
- Korean legal precedent corpus.
- The corpus consists of 150k cases.
- About 80k from [LAW OPEN DATA](https://www.law.go.kr/LSO/main.do) and 70k from LBox database.

- Example
```json
{
  "id": 99990,
  "precedent": "주문\n피고인을 징역 6개월에 처한다.\n다만, 이 판결 확정일로부터 1년간 위 형의 집행을 유예한다.\n\n이유\n범 죄 사 실\n1. 사기\n피고인은 2020. 12. 15. 16:00경 경북 칠곡군 B에 있는 피해자 C이 운영하는 ‘D’에서, 마치 정상적으로 대금을 지급할 것처럼 행세하면서 피해자에게 술을 주문하였다.\n그러나 사실 피고인은 수중에 충분한 현금이나 신용카드 등 결제 수단을 가지고 있지 않아 정상적으로 대금을 지급할 의사나 능력이 없었다.\n그럼에도 피고인은 위와 같이 피해자를 기망하여 이에 속은 피해자로부터 즉석에서 합계 8,000원 상당의 술을 교부받았다.\n2. 공무집행방해\n피고인은 제1항 기재 일시·장소에서, ‘손님이 술값을 지불하지 않고 있다’는 내용의 112신고를 접수하고 현장에 출동한 칠곡경찰서 E지구대 소속 경찰관 F로부터 술값을 지불하고 귀가할 것을 권유받자, “징역가고 싶은데 무전취식했으니 유치장에 넣어 달라”고 말하면서 순찰차에 타려고 하였다. 이에 경찰관들이 수회 귀가 할 것을 재차 종용하였으나, 피고인은 경찰관들을 향해 “내가 돌로 순찰차를 찍으면 징역갑니까?, 내여경 엉덩이 발로 차면 들어갈 수 있나?”라고 말하고, 이를 제지하는 F의 가슴을 팔꿈치로 수회 밀쳐 폭행하였다.\n이로써 피고인은 경찰관의 112신고사건 처리에 관한 정당한 직무집행을 방해하였다. 증거의 요지\n1. 피고인의 판시 제1의 사실에 부합하는 법정진술\n1. 증인 G, F에 대한 각 증인신문조서\n1. 영수증\n1. 현장 사진\n법령의 적용\n1. 범죄사실에 대한 해당법조 및 형의 선택\n형법 제347조 제1항, 제136조 제1항, 각 징역형 선택\n1. 경합범가중\n형법 제37조 전단, 제38조 제1항 제2호, 제50조\n1. 집행유예\n형법 제62조 제1항\n양형의 이유\n1. 법률상 처단형의 범위: 징역 1월∼15년\n2. 양형기준에 따른 권고형의 범위\n가. 제1범죄(사기)\n[유형의 결정]\n사기범죄 > 01. 일반사기 > [제1유형] 1억 원 미만\n[특별양형인자]\n- 감경요소: 미필적 고의로 기망행위를 저지른 경우 또는 기망행위의 정도가 약한 경우, 처벌불원\n[권고영역 및 권고형의 범위]\n특별감경영역, 징역 1월∼1년\n[일반양형인자] 없음\n나. 제2범죄(공무집행방해)\n[유형의 결정]\n공무집행방해범죄 > 01. 공무집행방해 > [제1유형] 공무집행방해/직무강요\n[특별양형인자]\n- 감경요소: 폭행·협박·위계의 정도가 경미한 경우\n[권고영역 및 권고형의 범위]\n감경영역, 징역 1월∼8월\n[일반양형인자]\n- 감경요소: 심신미약(본인 책임 있음)\n다. 다수범죄 처리기준에 따른 권고형의 범위: 징역 1월∼1년4월(제1범죄 상한 + 제2범죄 상한의 1/2)\n3. 선고형의 결정: 징역 6월에 집행유예 1년\n만취상태에서 식당에서 소란을 피웠고, 112신고로 출동한 경찰관이 여러 차례 귀가를 종용하였음에도 이를 거부하고 경찰관의 가슴을 밀친 점 등을 종합하면 죄책을 가볍게 볼 수 없으므로 징역형을 선택하되, 평소 주량보다 훨씬 많은 술을 마신 탓에 제정신을 가누지 못해 저지른 범행으로 보이고 폭행 정도가 매우 경미한 점, 피고인이 술이 깬 후 자신의 경솔한 언동을 깊이 반성하면서 재범하지 않기 위해 정신건강의학과의 치료 및 상담을 받고 있는 점, 식당 업주에게 피해를 변상하여 용서를 받은 점, 피고인의 나이와 가족관계 등의 사정을 참작하여 형의 집행을 유예하고, 범행 경위와 범행 후 피고인의 태도 등에 비추어 볼 때 재범의 위험성은 그다지 우려하지 않아도 될 것으로 보여 보호관찰 등 부수처분은 부과하지 않음.\n이상의 이유로 주문과 같이 판결한다."
}
```
- `id`: a data id.
- `precedent`: a case from the court of Korea. It includes the ruling (주문), the gist of claim (청구취지), the claim of appeal (항소취지), and
  the reasoning (이유).

### `casename_classification`

- Task: for the given facts (사실관계), a model is asked to predict the case name.
- The dataset consists of 10k `(facts, case name)` pairs extracted from Korean precedents.
- There are 100 classes (case categories) and each class contains 100 corresponding examples.
- 8,000 training, 1,000 validation, 1,000 test, and 1,294 test2 examples. The test2 set consists of examples that do not overlap with the precedents in `precedent_corpus`.

- Example

```json
{
  "id": 80,
  "casetype": "criminal",
  "casename": "감염병의예방및관리에관한법률위반",
  "facts": "질병관리청장, 시·도지사 또는 시장·군수·구청장은 제1급 감염병이 발생한 경우 감염병의 전파방지 및 예방을 위하여 감염병의심자를 적당한 장소에 일정한 기간 격리시키는 조치를 하여야 하고, 그 격리조치를 받은 사람은 이를 위반하여서는 아니 된다. 피고인은 해외에서 국내로 입국하였음을 이유로 2021. 4. 21.경 감염병의심자로 분류되었고, 같은 날 창녕군수로부터 ‘2021. 4. 21.부터 2021. 5. 5. 12:00경까지 피고인의 주거지인 경남 창녕군 B에서 격리해야 한다’는 내용의 자가격리 통지서를 수령하였다. 1. 2021. 4. 27.자 범행 그럼에도 불구하고 피고인은 2021. 4. 27. 11:20경에서 같은 날 11:59경까지 사이에 위 격리장소를 무단으로 이탈하여 자신의 승용차를 이용하여 경남 창녕군 C에 있는 ‘D’ 식당에 다녀오는 등 자가격리 조치를 위반하였다. 2. 2021. 5. 3.자 범행 피고인은 2021. 5. 3. 10:00경에서 같은 날 11:35경까지 사이에 위 격리장소를 무단으로 이탈하여 자신의 승용차를 이용하여 불상의 장소를 다녀오는 등 자가격리 조치를 위반하였다."
}
```
- `id`: a data id.
- `casetype`: a case type. The value is either `civil` (민사) or `criminal` (형사).
- `casename`: a case name.
- `facts`: facts (사실관계) extracted from `reasoning` (이유) section of individual cases.

### `statute_classification`

- Task: for a given facts (사실관계), a model is asked to predict related statutes (법령).
- The dataset consists of 2760 `(facts, statutes)` pairs extracted from individual Korean legal cases.
- There are 46 classes (case categories) and each class has 60 examples.
- 2,208 training, 276 validation, 276 test, 538 test2 examples. The test2 set consists of examples that do not overlap with the precedents in `precedent_corpus`.
- Example

```json
{
  "id": 5180,
  "casetype": "criminal",
  "casename": "사문서위조, 위조사문서행사",
  "statutes": [
    "형법 제231조",
    "형법 제234조"
  ],
  "facts": "1. 사문서위조 피고인은 2014. 5. 10.경 서울 송파구 또는 하남시 이하 알 수 없는 장소에서 영수증문구용지에 검정색 볼펜을 사용하여 수신인란에 ‘A’, 일금란에 ‘오천오백육십만원정’, 내역 란에 ‘2010가합7485사건의 합의금 및 피해 보상금 완결조’, 발행일란에 ‘2014년 5월 10일’이라고 기재한 뒤, 발행인 옆에 피고인이 임의로 만들었던 B의 도장을 찍었다. 이로써 피고인은 행사할 목적으로 사실증명에 관한 사문서인 B 명의의 영수증 1장을 위조하였다. 2. 위조사문서행사 피고인은 2014. 10. 16.경 하남시 이하 알 수 없는 장소에서 피고인이 B에 대한 채무를 모두 변제하였기 때문에 B가 C회사에 채권을 양도한 것을 인정할 수 없다는 취지의 내용증명원과 함께 위와 같이 위조한 영수증 사본을 마치 진정하게 성립한 문서인 것처럼 B에게 우편으로 보냈다. 이로써 피고인은 위조한 사문서를 행사하였다."
}

```

- `id`: a data id.
- `casetype`: a case type. The value is always `criminal`.
- `casename`: a case name.
- `statutes`: related statues.
- `facts`: facts (사실관계) extracted from `reasoning` (이유) section of individual cases.

### `ljp_criminal`

- Task: a model needs to predict the ranges of fine (벌금), imprisonment with labor (징역), imprisonment without labor (금고).
- 10,500 `facts` and the corresponding punishment are extracted from cases with following case categories are “indecent
  act by compulsion” (강제추행), “obstruction of performance of official duties” (공무집행방해), “bodily injuries from traffic
  accident” (교통사고처리특례법위반(치상)), “drunk driving” (도로교통 법위반(음주운전)), “fraud” (사기), “inflicting bodily injuries” (상해), and
  “violence” (폭행)
- 8,400 training, 1,050 validation, 1,050 test, 928 test2 examples. The test2 set consists of the examples from the test set that do not overlap with the precedents in `precedent_corpus`.
- Example
```json
{
  "casename": "공무집행방해",
  "casetype": "criminal",
  "facts": "피고인은 2020. 3. 13. 18:57경 수원시 장안구 B 앞 노상에서 지인인 C와 술을 마시던 중 C를 때려 112신고를 받고 출동한 수원중부경찰서 D지구대 소속 경위 E가 C의 진술을 청취하고 있는 모습을 보고 화가 나 '씨발,개새끼'라며 욕설을 하고, 위 E가 이를 제지하며 귀가를 종용하자 그의 왼쪽 뺨을 오른 주먹으로 1회 때려 폭행하였다.\n이로써 피고인은 경찰관의 112신고사건 처리에 관한 정당한 직무집행을 방해하였다. 증거의 요지\n1. 피고인의 법정진술\n1. 피고인에 대한 경찰 피의자신문조서\n1. E에 대한 경찰 진술조서\n1. 현장사진 등, 바디캠영상",
  "id": 2300,
  "label": {
    "fine_lv": 0,
    "imprisonment_with_labor_lv": 2,
    "imprisonment_without_labor_lv": 0,
    "text": "징역 6월"
  },
  "reason": "양형의 이유\n1. 법률상 처단형의 범위: 징역 1월∼5년\n2. 양형기준에 따른 권고형의 범위\n[유형의 결정]\n공무집행방해범죄 > 01. 공무집행방해 > [제1유형] 공무집행방해/직무강요\n[특별양형인자] 없음\n[권고영역 및 권고형의 범위] 기본영역, 징역 6월∼1년6월\n3. 선고형의 결정\n피고인이 싸움 발생 신고를 받고 출동한 경찰관에게 욕설을 퍼붓고 귀가를 종용한다는 이유로 경찰관의 뺨을 때리는 등 폭행을 행사하여 경찰관의 정당한 공무집행을 방해한 점에서 그 죄책이 매우 무겁다. 피고인의 범죄 전력도 상당히 많다.\n다만, 피고인이 범행을 인정하면서 반성하고 있는 점, 공무집행방해 범죄로 처벌받은 전력이 없는 점 등은 피고인에게 유리한 정상으로 참작한다.\n그 밖에 피고인의 연령, 성행, 환경, 가족관계, 건강상태, 범행의 동기와 수단 및 결과, 범행 후의 정황 등 이 사건 기록 및 변론에 나타난 모든 양형요소를 종합하여, 주문과 같이 형을 정한다.",
  "ruling": {
    "parse": {
      "fine": {
        "type": "",
        "unit": "",
        "value": -1
      },
      "imprisonment": {
        "type": "징역",
        "unit": "mo",
        "value": 6
      }
    },
    "text": "피고인을 징역 6월에 처한다.\n다만 이 판결 확정일로부터 2년간 위 형의 집행을 유예한다."
  }
}
```

- `id`: a data id.
- `casetype`: a case type. The value is always `criminal`.
- `casename`: a case name.
- `facts`: facts (사실관계) extracted from `reasoning` (이유) section of individual cases.
- `label`
    - `fine_lv`: a label representing individual ranges of the fine amount. See our paper for the detail.
    - `imprisonment_with_labor_lv`: a label representing the ranges of the imprisonemnt with labor.
    - `imprisonment_without_labor_lv`: a label for the imprisonment without labor case.
- `reason`: the reason for the punishment (양형의 이유).
- `ruling`: the ruling (주문) and its parsing result. `"" and -1` indicates null values.

### `ljp_civil`

- Task: a model is asked to predict the claim acceptance level (= "the approved money" / "the claimed money")
- 4,678 `facts` and the corresponding acceptance lv from 4 case categories: 929 examples from “price of
  indemnification” (구상금), 745 examples from “loan” (대여금), 1,004 examples from “unfair profits” (부당이득금), and 2,000
  examples from “lawsuit for damages (etc)” (손해배상(기)).
- 3,742 training, 467 validation, 467 test, 403 test2 examples. The test2 set consists of the test set examples those do not overlap with the precedents in `precedent_corpus`.
- Example
```json
{
  "id": 99,
  "casetype": "civil",
  "casename": "구상금",
  "claim_acceptance_lv": 1,
  "facts": "가. C는 2017. 7. 21. D으로부터 100,000,000원을 이율 연 25%, 변제기 2017. 8. 20.로 정하여 차용하였고(이하 ‘이 사건 차용금채무'라고 한다), 피고는 이 사건 차용금 채무를 보증한도액 140,000,000원, 보증기한 10년으로 정하여 연대보증하였으며, 같은 날 이 사건 차용금채무에 관한 공정증서를 작성하였다(공증인가 법무법인 E 증서 2017년 제392호, 이하 ‘이 사건 공정증서'라고 한다).\n나. 원고는 이 사건 차용금채무와 관련하여 원고 소유의 안산시 상록구 F, G, H 및 그 지상 건물(이하 ‘이 사건 부동산'이라고 한다)을 담보로 제공하기로 하여 2017. 7. 21. 수원지방법원 안산지원 접수 제53820호로 채권최고액 140,000,000원, 채무자 C, 근저당권자 D으로 한 근저당권설정등기를 경료하는 한편, 2018. 7. 13. D에게 이 사건 공정증서에 기한 채무를 2018. 7. 31.까지 변제하고, 변제기 이후 연 24%의 비율로 계산한 지연손해금을 지급하기로 하는 차용증을 작성하여 주었다(이하 ‘이 사건 차용증'이라고 한다).\n다. 원고는 2019. 11. 29. D에게 이 사건 차용금채무 원리금으로 합계 157,500,000원을 변제하였다.",
  "gist_of_claim": {
    "money": {
      "provider": "피고",
      "taker": "원고",
      "unit": "won",
      "value": 140000000
    },
    "text": "피고는 원고에게 140,000,000원 및 이에 대한 2019. 11. 30.부터 이 사건 소장 부본 송달일까지는 연 5%의, 그 다음날부터 다 갚는 날까지는 연 12%의 각 비율로 계산한 돈을 지급하라."
  },
  "ruling": {
    "litigation_cost": 0.5,
    "money": {
      "provider": "피고",
      "taker": "원고",
      "unit": "won",
      "value": 78750000
    },
    "text": "1. 피고는 원고에게 78,750,000원 및 이에 대한 2019. 11. 30.부터 2021. 11. 26.까지는 연 5%의, 그 다음날부터 다 갚는 날까지는 연 12%의 각 비율로 계산한 돈을 지급하라.\n2. 원고의 나머지 청구를 기각한다.\n3. 소송비용 중 1/2은 원고가 나머지는 피고가 각 부담한다.\n4. 제1항은 가집행할 수 있다."
  }
}

```

- `id`: a data id.
- `casetype`: a case type. The value is always `civil`.
- `casename`: a case name.
- `facts`: facts (사실관계) extracted from `reasoning` (이유) section of individual cases.
- `claim_acceptaance_lv`: the claim acceptance level. `0`, `1`, and `2` indicate rejection, partial approval, and full approval respectively.
- `gist_of_claim`: a gist of claim from plaintiffs (청구 취지) and its parsing result.
- `ruling`: a ruling (주문) and its parsing results.
  - `litigation_cost`: the ratio of the litigation cost that the plaintiff should pay.

### `summarization`

- Task: a model is asked to summarize precedents from the Supreme Court of Korea.
- The dataset is obtained from [LAW OPEN DATA](https://www.law.go.kr/LSO/main.do).
- The dataset consists of 20k `(precendent, summary)` pairs.
- 16,000 training, 2,000 validation, and 2,000 test examples.

- Example

```json
{
  "id": 16454,
  "summary": "[1] 피고와 제3자 사이에 있었던 민사소송의 확정판결의 존재를 넘어서 그 판결의 이유를 구성하는 사실관계들까지 법원에 현저한 사실로 볼 수는 없다. 민사재판에 있어서 이미 확정된 관련 민사사건의 판결에서 인정된 사실은 특별한 사정이 없는 한 유력한 증거가 되지만, 당해 민사재판에서 제출된 다른 증거 내용에 비추어 확정된 관련 민사사건 판결의 사실인정을 그대로 채용하기 어려운 경우에는 합리적인 이유를 설시하여 이를 배척할 수 있다는 법리도 그와 같이 확정된 민사판결 이유 중의 사실관계가 현저한 사실에 해당하지 않음을 전제로 한 것이다.\n\n\n[2] 원심이 다른 하급심판결의 이유 중 일부 사실관계에 관한 인정 사실을 그대로 인정하면서, 위 사정들이 ‘이 법원에 현저한 사실’이라고 본 사안에서, 당해 재판의 제1심 및 원심에서 다른 하급심판결의 판결문 등이 증거로 제출된 적이 없고, 당사자들도 이에 관하여 주장한 바가 없음에도 이를 ‘법원에 현저한 사실’로 본 원심판단에 법리오해의 잘못이 있다고 한 사례.",
  "precedent": "주문\n원심판결을 파기하고, 사건을 광주지방법원 본원 합의부에 환송한다.\n\n이유\n상고이유를 판단한다.\n1. 피고와 제3자 사이에 있었던 민사소송의 확정판결의 존재를 넘어서 그 판결의 이유를 구성하는 사실관계들까지 법원에 현저한 사실로 볼 수는 없다(대법원 2010. 1. 14. 선고 2009다69531 판결 참조). 민사재판에 있어서 이미 확정된 관련 민사사건의 판결에서 인정된 사실은 특별한 사정이 없는 한 유력한 증거가 되지만, 당해 민사재판에서 제출된 다른 증거 내용에 비추어 확정된 관련 민사사건 판결의 사실인정을 그대로 채용하기 어려운 경우에는 합리적인 이유를 설시하여 이를 배척할 수 있다는 법리(대법원 2018. 8. 30. 선고 2016다46338, 46345 판결 등 참조)도 그와 같이 확정된 민사판결 이유 중의 사실관계가 현저한 사실에 해당하지 않음을 전제로 한 것이다.\n2. 원심은 광주고등법원 2003나8816 판결 이유 중 ‘소외인이 피고 회사를 설립한 경위’에 관한 인정 사실, 광주지방법원 목포지원 2001가합1664 판결과 광주고등법원 2003나416 판결 이유 중 ‘피고 회사 이사회의 개최 여부’에 관한 인정 사실을 그대로 인정하면서, 위 사정들이 ‘이 법원에 현저한 사실’이라고 보았다.\n그런데 이 사건 기록에 의하면, 광주고등법원 2003나8816 판결, 광주지방법원 목포지원 2001가합1664 판결, 광주고등법원 2003나416 판결은 제1심 및 원심에서 판결문 등이 증거로 제출된 적이 없고, 당사자들도 이에 관하여 주장한 바가 없다.\n그렇다면 원심은 ‘법원에 현저한 사실’에 관한 법리를 오해한 나머지 필요한 심리를 다하지 아니한 채, 당사자가 증거로 제출하지 않고 심리가 되지 않았던 위 각 판결들에서 인정된 사실관계에 기하여 판단한 잘못이 있다. 이 점을 지적하는 상고이유 주장은 이유 있다.\n3. 그러므로 나머지 상고이유에 대한 판단을 생략한 채 원심판결을 파기하고, 사건을 다시 심리·판단하게 하기 위하여 원심법원에 환송하기로 하여, 관여 대법관의 일치된 의견으로 주문과 같이 판결한다."
}
```

- `id`: a data id.
- `summary`: a summary (판결요지) of given precedent (판결문).
- `precedent`: a case from the Korean supreme court.



# Models

## How to use the language model `lcube-base`
```python
# !pip instal transformers==4.19.4
import transformers

model = transformers.GPT2LMHeadModel.from_pretrained("lbox/lcube-base")
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "lbox/lcube-base",
    bos_token="[BOS]",
    unk_token="[UNK]",
    pad_token="[PAD]",
    mask_token="[MASK]",
)

text = "피고인은 불상지에 있는 커피숍에서, 피해자 B으로부터"
model_inputs = tokenizer(text,
                         max_length=1024,
                         padding=True,
                         truncation=True,
                         return_tensors='pt')
out = model.generate(
    model_inputs["input_ids"], 
    max_new_tokens=150,
    pad_token_id=tokenizer.pad_token_id,
    use_cache=True,
    repetition_penalty=1.2,
    top_k=5,
    top_p=0.9,
    temperature=1,
    num_beams=2,
)
tokenizer.batch_decode(out)
```

## Fine-tuning 
### Setup

```bash
conda create -n lbox-open pytyon=3.8.11
conda install pytorch==1.10.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

### Training

```bash
python run_model.py [TRINING_CONFIG_FILE_PATH] --mode train
````
See also  `scripts/train_[TASK].sh`

### Test

1. Make the test config file from the training config file by copying and changing the values of `trained` and `path` fields as shown below.
```yaml
train:
  weights:
    trained: true 
    path: ./models/[THE NAME OF THE TRAININ CONFIG FILE]/epoch=[XX]-step=[XX].ckpt
```
2.
```bash
python run_model.py [TEST_CONFIG_FILE_PATH] --mode test
````
See also  `scripts/test_[TASK].sh`



# Licensing Information

Copyright 2022-present [LBox Co. Ltd.](https://lbox.kr/)

Licensed under the [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)
