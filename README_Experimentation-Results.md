# Experimentation Results README
A guide on reading our [experimentation results](https://docs.google.com/spreadsheets/d/1rwfBER35Qyxlaptcd_pmgeVYZ_KmJ7rf1X7-l2080yw/edit?usp=sharing). The manner of data gathering we've done isn't quite the most readable and intuitive however we hope that this guide will help you better understand and explore our findings. We have also added annotations and descriptions in the sheets themselves, along with examples on how to read the data.

## Definition of Terms

Because we have adapted previous work by Casauay et al., we have also used their terms in this sheet. We have made some changes in them for our paper but regardless, the following terms are interchangeable:
- Binary Addition, Baseline category == `add-sm`
- Binary Addition, Original category == `add-lg`
- Binary Addition, Adversarial category == `add-adv`
- Binary Subtraction, Baseline category == `sub-sm`
- Binary Subtraction, Original 1 category == `sub1-lg`
- Binary Subtraction, Original 2 category == `sub3-lg`
- Binary Subtraction, Adversarial category == `sub-adv`

## Sheets

| Name | Purpose |
| ---- | ------- |
| COMPARE | Used as the main sheet for comparing values between CPU and GPU runs. |
| CPU | Contains all data related to sequential runs made on CPU. |
| GPU | Contains all data related to partially parallel runs made on CPU with GPU. |
| Evolutions | Contains runtime of evolutions made per run on different program variants (CPU, CPU+GPU) |
| GA_operations | Contains total runtimes of each GA operator per run. |
| AVERAGE | Tracks average of precision, runtime, etc. per variant. Important to note that this sheet is not entirely reliable due to losing precision from averaging averages. |
| csv_sub1-lg | As you may have read from our paper, binary subtraction in Original 1 category displayed the most finnicky results which is why we decided to add more runs for this category and this sheet contains information regarding it. |
