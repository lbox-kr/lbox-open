# LBox Open

A Legal AI Benchmark Dataset from Korean Legal Cases by [LBox](https://lbox.kr)

## Authors

- [Wonseok Hwang](mailto:wonseok.hwang@lbox.kr)
- [Dongjun Lee](mailto:dongjun.lee@lbox.kr)
- [Kyoungyeon Cho](mailto:kycho@lbox.kr)
- [Minjoon Seo](mailto:minjoon@lbox.kr)

## Updates
- Jun 2022: `lbox-open-v0.2` is coming soon!
- Mar 2022: We release `lbox-open-v0.1`!

## How to use the dataset

We use [`datasets` library](https://github.com/huggingface/datasets) from `Hugging Face`.
```python
# !pip install datasets
from datasets import load_dataset

# casename classficiation task
data_cn = load_dataset("lbox/lbox_open", "casename_classification")

# statutes classification task
data_st = load_dataset("lbox/lbox_open", "statute_classification")

# case summarization task
data_summ = load_dataset("lbox/lbox_open", "summarization")

# case corpus
data_corpus = load_dataset("lbox/lbox_open", "case_corpus")
```

## Tutorial
- [Explore the dataset on Colab](https://colab.research.google.com/drive/1CNkZzOfgOfhdJ-E6BYuv61iFMZSx62AU?usp=sharing)
- [Build a simple baseline model on Colab](https://colab.research.google.com/drive/1TSzlyc8cslM_0cP-TnB0YhnUMcNd4o1h?usp=sharing)
## Dataset Description

### `casename_classification`

- Task: for a given facts (사실관계), predict a case name.
- The dataset consists of 10k `(facts, case name)` pairs extracted from individual Korean legal cases.
- There are 100 classes (casenames).
- Each class has 100 corresponding examples.
- 8000 training, 1000 validation, and 1000 test examples.

#### Example

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

- Task: for a given facts (사실관계), predict related statutes (법령).
- The dataset consists of 2760 `(facts, statutes)` pairs extracted from individual Korean legal cases.
- There are 46 case name types and each type includes 60 examples.
- 2208 training, 276 validation, and 276 test examples.
#### Example

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
- `casetype`: a case type. The value is either `civil` (민사) or `criminal` (형사).
- `casename`: a case name.
- `statutes`: related statues.
- `facts`: facts (사실관계) extracted from `reasoning` (이유) section of individual cases.

### `summarization`

- Task: summarize cases from the supreme court of Korea.
- The dataset is obtained from [LAW OPEN DATA](https://www.law.go.kr/LSO/main.do).
- The dataset consists of 20k `(precendent, summary)` pairs.
- 16,000 training, 2,000 validation, and 2,000 test examples.

#### Example

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

### `case_corpus`

- Korean legal case corpus.
- The corpus consists of 150k cases.
- About 80k from [LAW OPEN DATA](https://www.law.go.kr/LSO/main.do) and 70k from LBox database.

#### Example

```json
{
  "id": 99990,
  "precedent": "주문\n피고인을 징역 6개월에 처한다.\n다만, 이 판결 확정일로부터 1년간 위 형의 집행을 유예한다.\n\n이유\n범 죄 사 실\n1. 사기\n피고인은 2020. 12. 15. 16:00경 경북 칠곡군 B에 있는 피해자 C이 운영하는 ‘D’에서, 마치 정상적으로 대금을 지급할 것처럼 행세하면서 피해자에게 술을 주문하였다.\n그러나 사실 피고인은 수중에 충분한 현금이나 신용카드 등 결제 수단을 가지고 있지 않아 정상적으로 대금을 지급할 의사나 능력이 없었다.\n그럼에도 피고인은 위와 같이 피해자를 기망하여 이에 속은 피해자로부터 즉석에서 합계 8,000원 상당의 술을 교부받았다.\n2. 공무집행방해\n피고인은 제1항 기재 일시·장소에서, ‘손님이 술값을 지불하지 않고 있다’는 내용의 112신고를 접수하고 현장에 출동한 칠곡경찰서 E지구대 소속 경찰관 F로부터 술값을 지불하고 귀가할 것을 권유받자, “징역가고 싶은데 무전취식했으니 유치장에 넣어 달라”고 말하면서 순찰차에 타려고 하였다. 이에 경찰관들이 수회 귀가 할 것을 재차 종용하였으나, 피고인은 경찰관들을 향해 “내가 돌로 순찰차를 찍으면 징역갑니까?, 내여경 엉덩이 발로 차면 들어갈 수 있나?”라고 말하고, 이를 제지하는 F의 가슴을 팔꿈치로 수회 밀쳐 폭행하였다.\n이로써 피고인은 경찰관의 112신고사건 처리에 관한 정당한 직무집행을 방해하였다. 증거의 요지\n1. 피고인의 판시 제1의 사실에 부합하는 법정진술\n1. 증인 G, F에 대한 각 증인신문조서\n1. 영수증\n1. 현장 사진\n법령의 적용\n1. 범죄사실에 대한 해당법조 및 형의 선택\n형법 제347조 제1항, 제136조 제1항, 각 징역형 선택\n1. 경합범가중\n형법 제37조 전단, 제38조 제1항 제2호, 제50조\n1. 집행유예\n형법 제62조 제1항\n양형의 이유\n1. 법률상 처단형의 범위: 징역 1월∼15년\n2. 양형기준에 따른 권고형의 범위\n가. 제1범죄(사기)\n[유형의 결정]\n사기범죄 > 01. 일반사기 > [제1유형] 1억 원 미만\n[특별양형인자]\n- 감경요소: 미필적 고의로 기망행위를 저지른 경우 또는 기망행위의 정도가 약한 경우, 처벌불원\n[권고영역 및 권고형의 범위]\n특별감경영역, 징역 1월∼1년\n[일반양형인자] 없음\n나. 제2범죄(공무집행방해)\n[유형의 결정]\n공무집행방해범죄 > 01. 공무집행방해 > [제1유형] 공무집행방해/직무강요\n[특별양형인자]\n- 감경요소: 폭행·협박·위계의 정도가 경미한 경우\n[권고영역 및 권고형의 범위]\n감경영역, 징역 1월∼8월\n[일반양형인자]\n- 감경요소: 심신미약(본인 책임 있음)\n다. 다수범죄 처리기준에 따른 권고형의 범위: 징역 1월∼1년4월(제1범죄 상한 + 제2범죄 상한의 1/2)\n3. 선고형의 결정: 징역 6월에 집행유예 1년\n만취상태에서 식당에서 소란을 피웠고, 112신고로 출동한 경찰관이 여러 차례 귀가를 종용하였음에도 이를 거부하고 경찰관의 가슴을 밀친 점 등을 종합하면 죄책을 가볍게 볼 수 없으므로 징역형을 선택하되, 평소 주량보다 훨씬 많은 술을 마신 탓에 제정신을 가누지 못해 저지른 범행으로 보이고 폭행 정도가 매우 경미한 점, 피고인이 술이 깬 후 자신의 경솔한 언동을 깊이 반성하면서 재범하지 않기 위해 정신건강의학과의 치료 및 상담을 받고 있는 점, 식당 업주에게 피해를 변상하여 용서를 받은 점, 피고인의 나이와 가족관계 등의 사정을 참작하여 형의 집행을 유예하고, 범행 경위와 범행 후 피고인의 태도 등에 비추어 볼 때 재범의 위험성은 그다지 우려하지 않아도 될 것으로 보여 보호관찰 등 부수처분은 부과하지 않음.\n이상의 이유로 주문과 같이 판결한다."
}
```

- `id`: a data id.
- `precedent`: a case from the court of Korea. It includes ruling (주문), claim (청구취지), claim of appeal (항소취지), and
  reasoning (이유).

## Licensing Information

Copyright 2022-present [LBox Co. Ltd.](https://lbox.kr/)

Licensed under the [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)
