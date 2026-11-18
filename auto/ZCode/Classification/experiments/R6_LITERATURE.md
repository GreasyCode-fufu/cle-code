# R6 Literature Review: Lightweight CNN for 3-class Gastric CLE Classification

Compiled 2026-11-08 to support the R6 design document (base model: torchvision MobileNetV3-Large; 3 classes: normal gastric body / benign = atrophy + intestinal metaplasia (IM) / malignant = dysplasia + cancer).

## Verification policy and legend

- **Verified: Y** means the paper's existence and the facts quoted here were checked on a primary page: PubMed record (fetched through NCBI E-utilities), PMC full text, the CVF Open Access page or PDF, the official GitHub repo or README, or the HuggingFace model card or API.
- **Verified: Y (sec.)** means the paper's existence was verified on PubMed or CVF, but the detailed criteria quoted come from a verified *secondary* source (a review in PMC). The original full text was paywalled.
- **UNVERIFIED** means this environment could not reach a primary page. Blocked sources included arXiv, ECVA (ECCV), NeurIPS proceedings, OpenReview, ScienceDirect and wjgnet.com. Do not cite numbers marked UNVERIFIED without checking them yourself.
- Warning: the BibTeX blocks on some timm HuggingFace model cards contain **wrong author lists**. This applies to StarNet and GhostNetV3. Take author lists from CVF/arXiv/PubMed, not from timm cards (see Section 4).

---

## 0. Executive summary (design-relevant facts)

1. **No international consensus defines gastric CLE criteria.** The 2015 pCLE consensus (Wang KK et al.) covers Barrett's esophagus, biliary strictures, colorectal lesions and IBD only. Gastric criteria come from single-centre classifications, mostly from the Qilu Hospital (Shandong University) group: Zhang 2008 (eCLE, 7 pit types A–G), Guo 2008 (IM), Li WB 2011 (two-tiered cancer/HGIN), and Li Z 2016 (pCLE types I, IIa–c, IIIa–c plus vessel types 1–3). The Mainz classification (Kiesslich 2004/2006) was defined for **colon and Barrett's**, not stomach.
2. **The criteria are structural, which matches our hypothesis.**
   - Normal: regular pits with round openings and a honeycomb capillary network (corpus).
   - Atrophy: fewer pits with dilated openings.
   - IM: villiform pits plus dark goblet cells (mucin does not take up fluorescein), absorptive cells and brush border.
   - Neoplasia: irregular gland size and shape, distorted, disorganised or destroyed pits, dark irregular thickened epithelium, irregular dark cells, and irregular vessels with heterogeneous fluorescein leakage.
3. **The hardest boundary is LGIN, which can fall on either side.** Li 2016 type IIIa ("mild to moderate irregular pits") borders benign mucosa. Studies also disagree on where LGIN belongs: CCADS 2025 counts it as neoplasm, while the Pan 2026 training system counts LGIN as *non-cancerous*. Our "dysplasia → malignant" label rule must state explicitly where LGIN goes.
4. **The closest prior gastric CLE deep-learning work is CCADS** (Liu G et al., BMC Med 2025; Qilu group). It used Inception-ResNet-V2 on 58,901 Cellvizio pCLE images in 5 classes, with labels following Li 2016. It reported image-test accuracy of 95.05–99.61% and prospective real-time accuracy of 91.71–97.13%. **No gastric CLE work we found uses a lightweight or mobile backbone or an explicit structure prior.** That is the gap R6 fills.
5. **MatchED (CVPR 2026) is a crisp edge-*detection* paper**, by Cetinkaya, Kalkan and Akbas (METU). It is a ~21K-parameter plug-in that does one-to-one bipartite matching (linear sum assignment) between predicted and ground-truth edge pixels, then trains with BCE against the matched target. **It contains no "soft-chamfer" loss.** Our soft-chamfer gland-boundary loss is our own relaxation. Cite chamfer and distance-transform boundary losses for it (Section 3.3).
6. **The cross-scale star head (B) has a direct prior: Hierarchical Bilinear Pooling (ECCV 2018).** HBP already does cross-layer Hadamard-product bilinear pooling. MLB (ICLR 2017) and MFB (ICCV 2017) are the low-rank Hadamard bilinear forms, and StarNet's star operation is the same-input special case. Frame (B)'s novelty as the CLE-specific fine×coarse gland-scale evidence head, not as the operator itself.
7. **LSNet's official SKA is a Triton kernel**, so it needs CUDA and Triton and is not pure PyTorch. Every other candidate backbone in Section 4 loads from timm in pure PyTorch.

---

## 1. Clinical CLE diagnostic criteria for the stomach (motivation citations)

### 1.1 Mainz classification (Kiesslich et al.), defined for colon and Barrett's

- **[Kiesslich2004]** Kiesslich R, Burg J, Vieth M, Gnaendiger J, Enders M, Delaney P, Polglase A, McLaren W, Janell D, Thomas S, Nafe B, Galle PR, Neurath MF. *Confocal laser endoscopy for diagnosing intraepithelial neoplasias and colorectal cancer in vivo.* Gastroenterology. 2004;127(3):706-713. doi:10.1053/j.gastro.2004.06.050. PMID 15362025. **Verified: Y (PubMed).**
  - Defines the eCLE grading by cellular and vascular changes, later called the "Mainz classification". It was assessed on 13,020 images from 390 locations against 1,038 biopsies: sensitivity 97.4%, specificity 99.4%, accuracy 99.2% for neoplasia.
  - The criteria table comes from [Ussui2012] below. **Verified: Y (sec.)**
    - Normal: hexagonal, honeycomb vessels; regular luminal crypt openings; uniform epithelial layer.
    - Regeneration: star-shaped luminal crypt openings or clustered crypts; regular or fewer goblet cells.
    - Neoplasia: dilated and distorted vessels; ridged-lined irregular epithelial layer with loss of goblet cells, irregular cells and little mucin.
  - Relevance: this is the template for "regular versus irregular crypt/gland architecture plus vessels". Cite it as the origin of CLE pattern grading, but **not** as a gastric criterion.
- **[Kiesslich2006]** Kiesslich R, Gossner L, Goetz M, Dahlmann A, Vieth M, Stolte M, Hoffman A, Jung M, Nafe B, Galle PR, Neurath MF. *In vivo histology of Barrett's esophagus and associated neoplasia by confocal laser endomicroscopy.* Clin Gastroenterol Hepatol. 2006;4(8):979-987. doi:10.1016/j.cgh.2006.05.010. PMID 16843068. **Verified: Y (PubMed).**
  - "Confocal Barrett classification" in 63 patients. Barrett's: sensitivity 98.1%, specificity 94.1%. Neoplasia: sensitivity 92.9%, specificity 98.4%. Interobserver κ 0.843; intraobserver κ 0.892.
- **[Ussui2012]** Ussui VM, Wallace MB. *Confocal endomicroscopy of colorectal polyps.* Gastroenterol Res Pract. 2012;2012:545679. doi:10.1155/2012/545679. PMC3272798. **Verified: Y (PMC).** Secondary source that reproduces the Mainz table (Table 1).
- **[Kiesslich2005]** Kiesslich R, Goetz M, Burg J, et al. *Diagnosing Helicobacter pylori in vivo by confocal laser endoscopy.* Gastroenterology. 2005;128(7):2119-2123. doi:10.1053/j.gastro.2004.12.035. PMID 15940642. **Verified: Y (PubMed).** Early gastric use of eCLE by the Mainz group (case report). Not a classification.

### 1.2 Zhang et al. 2008: eCLE gastric pit-pattern classification (7 types)

- **[Zhang2008]** Zhang JN, Li YQ, Zhao YA, Yu T, Zhang JP, Guo YT, Liu H. *Classification of gastric pit patterns by confocal endomicroscopy.* Gastrointest Endosc. 2008;67(6):843-853. doi:10.1016/j.gie.2008.01.036. PMID 18440377. **Verified: Y (PubMed abstract)**, with type E/F detail **Y (sec.)** from [Han2023] and [Wang2010].
  - Cohort: 132 patients, 7 healthy volunteers and 10 ex-vivo cancer specimens; Pentax eCLE; blinded comparison with histology.

| Type | CLE appearance | Histology | Source |
|---|---|---|---|
| A | round pits with round openings | normal mucosa with fundic glands (**corpus/body, our "normal" class**) | abstract; [Han2023] |
| B | non-continuous short rod-like pits | corporal mucosa with gastritis | abstract |
| C | continuous short rod-like pits | normal pyloric (antral) mucosa | abstract |
| D | elongated, tortuous, branch-like pits | antral mucosa with gastritis | abstract |
| E | fewer gastric pits with markedly dilated calibre/opening | **atrophy**: sensitivity 83.6%, specificity 99.6% | abstract; [Han2023]; [Wang2010] |
| F | villous-like pits; dark goblet cells easily distinguished | **intestinal metaplasia** | abstract; [Han2023] |
| G | (cancer) split into poorly differentiated vs. differentiated adenocarcinoma according to whether diffusely atypical cells are present | **gastric cancer**: sensitivity 90.0%, specificity 99.4% | abstract; [Han2023] |

  - The verbatim wording of type G and its subtype names could not be retrieved (full text paywalled). **UNVERIFIED** beyond the paraphrase above.
  - Relevance: the canonical evidence that atrophy (sparse, dilated pits), IM (villiform pits with goblet cells) and cancer (loss of regular pits, atypical cells) have distinct structural signatures in CLE.

### 1.3 Qilu-group criteria for GIM, atrophy and neoplasia (eCLE 2008–2011; pCLE 2014–2017)

- **[Guo2008]** Guo YT, Li YQ, Yu T, Zhang TG, Zhang JN, Liu H, Liu FG, Xie XJ, Zhu Q, Zhao YA. *Diagnosis of gastric intestinal metaplasia with confocal laser endomicroscopy in vivo: a prospective study.* Endoscopy. 2008;40(7):547-553. doi:10.1055/s-2007-995633. PMID 18618938. **Verified: Y (PubMed).**
  - GIM is diagnosed if **any** of the following is present: goblet cells; columnar absorptive cells with brush border; villiform foveolar epithelium.
  - Complete vs. incomplete GIM is separated by goblet-cell shape, absorptive cells/brush border, and vessel and crypt architecture.
  - CLE vs. conventional endoscopy for GIM: sensitivity 98.13% vs. 36.88%, specificity 95.33% vs. 91.59%, κ 0.94.
- **[Wang2010]** Wang P, Ji R, Yu T, Zuo XL, Zhou CJ, Li CQ, Li Z, Li YQ. *Classification of histological severity of Helicobacter pylori-associated gastritis by confocal laser endomicroscopy.* World J Gastroenterol. 2010;16(41):5203-5210. doi:10.3748/wjg.v16.i41.5203. PMC2975091. **Verified: Y (PMC).**
  - Atrophy is "decreased gastric pits with a dilated opening".
  - IM is "villous-like gastric pits with goblet cells, absorptive cells and brush border".
  - Active inflammation is graded by increasing pit distortion, epithelial destruction and fluorescein leakage.
  - Relevance: an explicit statement that inflammation (part of our normal/benign spectrum) adds distortion and leakage without loss of the pit lattice.
- **[Li2010]** Li Z, Yu T, Zuo XL, Gu XM, Zhou CJ, Ji R, Li CQ, Wang P, Zhang TG, Ho KY, Li YQ. *Confocal laser endomicroscopy for in vivo diagnosis of gastric intraepithelial neoplasia: a feasibility study.* Gastrointest Endosc. 2010;72(6):1146-1153. doi:10.1016/j.gie.2010.08.031. PMID 21111868. **Verified: Y (PubMed).**
  - Introduces an "intraepithelial neoplasia score". For GIN: sensitivity 77.8%, specificity 81.8%, κ 0.70.
  - A score ≥5 separates HGIN from LGIN with sensitivity 66.7% and specificity 88.0%. This shows **dysplasia grading is hard even for experts**.
- **[LiWB2011]** Li WB, Zuo XL, Li CQ, Zuo F, Gu XM, Yu T, Chu CL, Zhang TG, Li YQ. *Diagnostic value of confocal laser endomicroscopy for gastric superficial cancerous lesions.* Gut. 2011;60(3):299-306. doi:10.1136/gut.2010.223586. PMID 21193460. **Verified: Y (PubMed)**; criteria **Y (sec.)** from [Han2023].
  - Two-tiered classification: non-cancerous vs. cancer/HGIN, based on architecture, cells and microvessels.
  - Cancer/HGIN criteria: "glandular size and shape are irregular, pits and glands are disorganised or destroyed, the cells become disordered, stratified, and lose polarity", plus irregular microvessel shape and calibre.
  - Phase II, 1,786 patients, real-time iCLE vs. white-light endoscopy: sensitivity 88.9% vs. 72.2%, specificity 99.3% vs. 95.1%, accuracy 98.8% vs. 94.1%.
  - Relevance: **the core citation for "fused/irregular/disordered glands = malignant".**
- **[LiZ2014]** Li Z, Zuo XL, Yu T, Gu XM, Zhou CJ, Li CQ, Ji R, Li YQ. *Confocal laser endomicroscopy for in vivo detection of gastric intestinal metaplasia: a randomized controlled trial.* Endoscopy. 2014;46(4):282-290. doi:10.1055/s-0033-1359215. PMID 24473908. **Verified: Y (PubMed).** RCT with 168 patients. Per-biopsy GIM yield was 65.70% with CLE-targeted biopsy vs. 15.73% with WLE standard biopsy, and CLE reduced the number of biopsies by 68%.
- **[LiZ2016]** Li Z, Zuo XL, Li CQ, Liu ZY, Ji R, Liu J, Guo J, Li YQ. *New classification of gastric pit patterns and vessel architecture using probe-based confocal laser endomicroscopy.* J Clin Gastroenterol. 2016;50(1):23-32. doi:10.1097/MCG.0000000000000298. PMID 25751373. **Verified: Y (PubMed)**; type definitions **Y (sec.)** from [Pilonis2022] and [Han2023].
  - Pit types:
    - **I**: regular pits with round/wide/slit-like openings (normal cardia/corpus/antrum).
    - **IIa**: regular pits with elongated openings and increased stromal fluorescence (inflammation).
    - **IIb**: reduced pits with dilated openings (**atrophy**).
    - **IIc**: goblet cells with dark mucin (**IM**).
    - **IIIa**: mild to moderate irregular pits with variable width of the epithelial lining (**LGIN**).
    - **IIIb**: prominent distorted pits with irregular epithelial lining (**HGIN**).
    - **IIIc**: atypical glands or dispersion of irregular dark cells (**adenocarcinoma**).
  - Vessel types:
    - Normal: regular capillaries of normal calibre (anfractuous in cardia, **honeycomb-like in corpus**, coil-shaped in antrum).
    - Inflamed: increased capillaries with elevated leakage.
    - Neoplastic: irregular capillaries with heterogeneous leakage or dilated calibre.
  - Performance:
    - IIb for atrophy: sensitivity 88.51%, specificity 99.19%.
    - IIc for IM: sensitivity 92.34%, specificity 99.34%.
    - Type 3 pit or vessel pattern for neoplasia: sensitivity 89.89%, specificity 99.44%.
    - Interobserver κ = 0.70.
  - Relevance: **the label standard used by CCADS 2025.** It is also the most direct mapping onto our classes: I/IIa → normal; IIb/IIc → benign; IIIa–c → malignant.
- **[Zuo2017]** Zuo XL, Li Z, Li CQ, Zheng YY, Xu LD, Chen J, Lin R, Song J, Yu CH, Yue M, Zhou Q, Liu ZY, Li YQ. *Probe-based endomicroscopy for in vivo detection of gastric intestinal metaplasia and neoplasia: a multicenter randomized controlled trial.* Endoscopy. 2017;49(11):1033-1042. doi:10.1055/s-0043-115382. PMID 28753702. **Verified: Y (PubMed).** 238 patients at 4 centres. FICE-guided pCLE raised per-biopsy yield of GIM/GIN/EGC from 31.5% to 75.1% and needed 48.5% fewer biopsies.
- **[Bok2013]** Bok GH, Jeon SR, Cho JY, et al. *The accuracy of probe-based confocal endomicroscopy versus conventional endoscopic biopsies for the diagnosis of superficial gastric neoplasia (with videos).* Gastrointest Endosc. 2013;77(6):899-908. doi:10.1016/j.gie.2013.01.018. PMID 23473002. **Verified: Y (PubMed).** 54 lesions. Agreement with final histology: in-vivo pCLE κ 0.824 vs. forceps biopsy κ 0.617. Adenocarcinoma accuracy 91.7% vs. 85.2%.
- **[Kitabatake2006]** Kitabatake S, Niwa Y, Miyahara R, et al. *Confocal endomicroscopy for the diagnosis of gastric cancer in vivo.* Endoscopy. 2006;38(11):1110-1114. doi:10.1055/s-2006-944855. PMID 17111332. **Verified: Y (PubMed).** 27 early gastric cancer patients. Pathologist accuracy 94.2% and 96.2%, but it "decreased substantially when poor images ... were included". This supports adding an image-quality caveat.

### 1.4 Miami classification for pCLE (gastric part)

- **[Wallace2011]** Wallace M, Lauwers GY, Chen Y, Dekker E, Fockens P, Sharma P, Meining A. *Miami classification for probe-based confocal laser endomicroscopy.* Endoscopy. 2011;43(10):882-891. doi:10.1055/s-0030-1256632. PMID 21818734. **Verified: Y (PubMed)**; gastric content **Y (sec.)** from [Han2023].
  - Gastric dysplasia: "an irregular crypt lumen and dark, irregular, thickened epithelium".
  - Early gastric cancer: "a completely disorganised epithelium, fluorescein leakage, and dark irregular epithelium".
  - Consensus of pCLE users reached at a February 2009 meeting.

### 1.5 Recent reviews, meta-analyses and consensus (2015–2026)

- **[WangKK2015]** Wang KK, Carr-Locke DL, Singh SK, et al. *Use of probe-based confocal laser endomicroscopy (pCLE) in gastrointestinal applications. A consensus report based on clinical evidence.* United European Gastroenterol J. 2015;3(3):230-254. doi:10.1177/2050640614566066. PMC4480534. **Verified: Y (PMC).**
  - Covers Barrett's, biliary strictures, colorectal lesions and IBD. **It contains no gastric statements.** Cite it to say that gastric CLE lacks an international consensus.
- **[Bai2017]** Bai T, Zhang L, Sharma S, Jiang YD, Xia J, Wang H, Qian W, Song J, Hou XH. *Diagnostic performance of confocal laser endomicroscopy for atrophy and gastric intestinal metaplasia: a meta-analysis.* J Dig Dis. 2017;18(5):273-282. doi:10.1111/1751-2980.12470. PMID 28342261. **Verified: Y (PubMed).** 10 studies. Atrophy: pooled sensitivity 88%, specificity 98%. GIM: 93% and 98%. SROC AUC 0.9491 (atrophy) and 0.9812 (GIM).
- **[Pilonis2022]** Pilonis ND, Januszewicz W, di Pietro M. *Confocal laser endomicroscopy in gastro-intestinal endoscopy: technical aspects and clinical applications.* Transl Gastroenterol Hepatol. 2022;7:7. doi:10.21037/tgh.2020.04.02. PMC8826043. **Verified: Y (PMC).** Reproduces the Li 2016 pCLE type definitions quoted above.
- **[Canakis2022]** Canakis A, Deliwala SS, Kadiyala J, Bomman S, Canakis J, Bilal M. *The diagnostic performance of probe-based confocal laser endomicroscopy in the detection of gastric cancer: a systematic review and meta-analysis.* Ann Gastroenterol. 2022;35(5):496-502. doi:10.20524/aog.2022.0741. PMC9399575. **Verified: Y (PubMed).** 7 studies, 567 patients. Pooled sensitivity 87.9%, specificity 96.5%, accuracy 94.7%.
- **[Han2023]** Han W, Kong R, Wang N, Bao W, Mao X, Lu J. *Confocal laser endomicroscopy for detection of early upper gastrointestinal cancer.* Cancers (Basel). 2023;15(3):776. doi:10.3390/cancers15030776. PMC9913498. **Verified: Y (PMC).** Most complete open-access summary of Zhang 2008, Li WB 2011, Li Z 2016 and the Miami gastric criteria, with a sensitivity/specificity table (its Table 3).
- **[LeeJY2026]** Lee JY, Lee SH, Ji YG, Jung HY, Koo JS, Park JM. *Confocal laser endomicroscopy for upper gastrointestinal neoplasia: systematic review and meta-analysis.* Endosc Int Open. 2026;14:a28631407. doi:10.1055/a-2863-1407. PMC13289976. **Verified: Y (PubMed).** 33 studies (16 gastric), 2,350 patients. Gastric neoplasia: pooled sensitivity 0.89 (0.83–0.93), specificity 0.95 (0.92–0.96). The authors warn of possible small-study effects.
- **[Ruan2026]** Ruan W, Yang J, Hou Y, Yv A, Ding Q, Zhou Y. *Clinical application of confocal laser endomicroscopy in the diagnosis of gastrointestinal diseases.* Digestion. 2026;107(5):436-448. doi:10.1159/000548414. PMID 40966172. **Verified: Y (PubMed).** Recent review that includes CLE plus AI.
- **[Pan2026]** Pan P, Gao J, Zhang Y, et al. (Bai Y, senior author). *Establishment and evaluation of a probe-based confocal laser endomicroscopy training system for gastric cancerous lesions diagnosis: evidence from a national training program.* Clin Transl Gastroenterol. 2026;17(9):e01074. doi:10.14309/ctg.0000000000001074. PMC13593169. **Verified: Y (PMC).**
  - The criteria are "based on the Miami diagnostic standard at a macro level and ... the classification of gastric pit patterns and vessel architecture in detail" (Li 2016, Li 2011).
  - Observation points: number, size and shape of glands; uniformity of glandular epithelial cells; fluorescein extravasation; brush border; goblet cells and dark clumps; microvessel shape, number and diameter.
  - Groups **LGIN with the non-cancerous lesions**.
  - 150 lesions, 127 endoscopists.

### 1.6 Synthesis: structural cues per class, mapped to our 3 classes

| Our class | Histology | Glands / pits (shape, density) | Arrangement / regularity | Epithelium / cells | Vessels / leakage | Key refs |
|---|---|---|---|---|---|---|
| Normal (body) | normal fundic/corpus mucosa | round pits, round openings (Zhang A; Li I) | regular, evenly spaced lattice | uniform epithelial lining | regular capillaries, honeycomb-like (corpus) | Zhang2008, LiZ2016 |
| (normal/benign border) | chronic/active gastritis | short rod-like or elongated openings (Zhang B/D; Li IIa) | mildly distorted, still regular | intact or partly damaged epithelium | more capillaries; focal → widespread leakage | Wang2010, LiZ2016 |
| Benign | atrophy | **fewer pits, dilated openings** (Zhang E; Li IIb) | sparse lattice | — | — | Zhang2008, Wang2010, LiZ2016 |
| Benign | intestinal metaplasia | **villiform/villous-like pits** (Zhang F) | regular, equidistant | **dark goblet cells** (mucin not stained), absorptive cells, brush border (Li IIc) | — | Guo2008, Wang2010, LiZ2016 |
| Malignant | LGIN (dysplasia) | **mild–moderate irregular pits**, variable epithelial width (Li IIIa) | starting to become irregular | dark, irregular, thickened epithelium (Miami) | may be irregular | LiZ2016, Wallace2011 |
| Malignant | HGIN | **prominently distorted pits** (Li IIIb) | disorganised | irregular epithelial lining; cells disordered, stratified, loss of polarity | irregular, dilated | LiWB2011, LiZ2016 |
| Malignant | adenocarcinoma | **atypical glands; destroyed or absent pits** (Zhang G; Li IIIc) | completely disorganised | dispersed irregular dark cells (poorly differentiated) | heterogeneous leakage, irregular calibre | Zhang2008, LiWB2011, LiZ2016, Wallace2011 |

Design implications. Items 1, 3 and 4 below are our own reading of the criteria above.

1. Benign is **bimodal**. Atrophy is a sparse or dilated lattice; IM is villiform with small dark round goblet inclusions. A single "regularity" feature will not capture both, so mid-level (cell-scale) and gland-scale features are both needed. This motivates the fine×coarse interaction in (B).
2. Malignant is defined by **local heterogeneity and irregular gland boundaries**: variable width, distortion, fusion and destruction. This motivates the dynamic local aggregation in (A) and the gland-boundary auxiliary in (C).
3. Vessel and leakage cues exist. We do not model them explicitly, but Grad-CAM may land on them.
4. The LGIN placement must be stated explicitly in the paper (see Executive summary, item 3).

---

## 2. Deep learning for CLE/pCLE images (2017–2026)

### 2.1 Gastric

| Key | Citation | Data / device | Classes | Model | Reported performance | Verified |
|---|---|---|---|---|---|---|
| **[Liu2025-CCADS]** | Liu G, Li G, Li Z, Shao X, Ji R, Ma T, Zhang Y, Su J, Qi Q, Guo J, He Y, Yang X, Li Y, Zuo X. *Deep learning-aided optical biopsy achieves whole-chain diagnosis of Correa cascade of gastric cancer: a prospective study.* BMC Med. 2025;23(1):527. doi:10.1186/s12916-025-04310-9. PMC12481803 | Cellvizio pCLE; 5,771 exams; 58,901 images (41,743 train / 5,719 val / 11,439 test, patient-disjoint) + 461 training videos; prospective cohort of 951 patients / 1,254 lesions | 5 classes (inflammation, atrophy, GIM, LGIN, HGIN/CA) + a quality gate | 2-layer CNN quality filter → **Inception-ResNet-V2** image classifier → 4-layer fully connected video aggregator; labels per Li 2016 | Image test accuracy 95.05–99.61%, AUC 0.9857–0.9998. Real-time prospective accuracy 91.71–97.13%. HGIN/CA sensitivity 98.44%, specificity 97.06%. Neoplasm sensitivity 96.70% vs. experts 89.01% | Y (PubMed + PMC) |
| **[Cho2024]** | Cho H, Moon D, Heo SM, Chu J, Bae H, et al. *Artificial intelligence-based real-time histopathology of gastric cancer using confocal laser endomicroscopy.* NPJ Precis Oncol. 2024;8(1):131. doi:10.1038/s41698-024-00621-x. PMC11178780 | **ex-vivo** handheld CLE (VPIX cCeLL-A 488); 43 fresh gastric cancer specimens; 7,480 tumour + 12,928 normal images; 1024×1024, 500×500 µm field of view | 2-stage: tumour vs. normal, then adenocarcinoma vs. non-adenocarcinoma | EfficientNetV2-S (~22M parameters), 480×480 input | Internal: CNN1 accuracy 0.964, CNN2 accuracy 0.990. External: accuracy 0.990. 0.043 s per image | Y (PMC) |
| **[Zhou2023-MixSiam]** | Zhou J, Dong X, Liu Q. *Boosting few-shot confocal endomicroscopy image recognition with feature-level MixSiam.* Biomed Opt Express. 2023;14(3):1054-1070. doi:10.1364/BOE.478832. PMC10026589 | Preclinical pCLE system; SSL_pCLE dataset: 2,369 images from 86 patients | 6 gastric/GI classes, incl. atrophic gastritis (568), normal gastric body (325), normal sinus (627), intestinal epithelial hyperplasia (405), "low-/high-level" lesions (277/167) | ResNet-18, self-supervised then few-shot | Linear evaluation top-1 74.33–76.82% vs. supervised 72.44–74.36%. 5-way 1-shot 63.62%, 5-shot 74.37% | Y (PMC) |
| **[Zhou2024-CTCL]** | Zhou J, Dong X, Liu Q. *Clustering-guided twin contrastive learning for endomicroscopy image classification.* IEEE J Biomed Health Inform. 2024;28(5):2879-2890. doi:10.1109/JBHI.2024.3366223 | pCLE GI tumour images | multi-stage GI tumour | contrastive self-supervised learning; **treats images at any rotation angle and with different noise as the same instance** (geometric invariance of CLE) | Linear top-1 75.60% / 78.45%; few-shot 64.12% / 77.37% | Y (PubMed) |
| **[Su2023]** | Su D, Zheng X, Wang S, Qi Q, Li Z. *Goblet cells segmentation from confocal laser endomicroscopy with an improved U-Net.* Biomed Phys Eng Express. 2023;9(5). doi:10.1088/2057-1976/ace89e | 343 CLE images from 62 patients | goblet-cell segmentation (GIM) | U-Net + pixel-gradient attention | IoU 87.95%, Dice 86.64% | Y (PubMed) |

Relevance:
- CCADS is the strongest baseline to position against. It uses a heavy backbone, 5 classes and a patient-disjoint split. It does not report parameters or FLOPs, does not target lightweight models, and has no structure prior.
- Cho 2024 is ex-vivo, so it is a different domain.
- CTCL supports treating CLE as rotation-invariant (isotropic texture), which justifies rotation augmentation and argues against axis-aligned operators.
- Su 2023 shows goblet cells (the IM cue) are segmentable structures.

### 2.2 Other organs

| Key | Citation | Organ / data | Classes | Model | Performance | Verified |
|---|---|---|---|---|---|---|
| [Aubreville2017] | Aubreville M, Knipfer C, Oetter N, Jaremenko C, Rodner E, Denzler J, Bohr C, Neumann H, Stelzle F, Maier A. *Automatic classification of cancerous tissue in laserendomicroscopy images of the oral cavity using deep learning.* Sci Rep. 2017;7(1):11979. doi:10.1038/s41598-017-12320-8 | Oral cavity; 7,894 CLE images | OSCC vs. normal | CNN (patch-based), compared against texture features | AUC 0.96, accuracy 88.3% (sensitivity 86.6%, specificity 90%) | Y (PubMed) |
| [Hong2017] | Hong J, Park BY, Park H. *Convolutional neural network classifier for distinguishing Barrett's esophagus and neoplasia endomicroscopy images.* Proc IEEE EMBC. 2017:2892-2895. doi:10.1109/EMBC.2017.8037461 | Barrett's; ISBI 2016 challenge; 262 images | IM / gastric metaplasia / neoplasia | small CNN | accuracy 80.77% (26 test images) | Y (PubMed) |
| [Tong2019] | Tong L, Wu H, Wang MD. *CAESNet: convolutional autoencoder based semi-supervised network for improving multiclass classification of endomicroscopic images.* J Am Med Inform Assoc. 2019;26(11):1286-1296. doi:10.1093/jamia/ocz089 | Barrett's eCLE (Emory); 429 labelled + 2,826 unlabelled | multiclass dysplasia grade | semi-supervised convolutional autoencoder | beats supervised CNN (see paper) | Y (PubMed) |
| [Guleria2021] | Guleria S, Shah TU, Pulido JV, et al. *Deep learning systems detect dysplasia with human-like accuracy using histopathology and probe-based confocal laser endomicroscopy.* Sci Rep. 2021;11(1):5086. doi:10.1038/s41598-021-84510-4 | Barrett's pCLE; 1,970 videos | squamous / non-dysplastic BE / dysplasia-cancer (3-class) | CNN + Grad-CAM | accuracy 90%; dysplasia sensitivity 71% | Y (PubMed) |
| [Udristoiu2021] | Udristoiu AL, Stefanescu D, Gruionu G, et al. *Deep learning algorithm for the confirmation of mucosal healing in Crohn's disease, based on confocal laser endomicroscopy images.* J Gastrointestin Liver Dis. 2021;30(1):59-65. doi:10.15403/jgld-3212 | Colon/ileum; 6,205 images from 54 patients | active Crohn's disease vs. healed/normal | CNN-LSTM (4 conv + 1 LSTM layer) | accuracy 95.3%, AUC 0.98. **Image-level 80/20 split within patients (leakage risk)** | Y (PubMed) |
| [Ziebart2021] | Ziebart A, Stadniczuk D, Roos V, Ratliff M, von Deimling A, Hänggi D, Enders F. *Deep neural network for differentiation of brain tumor tissue displayed by confocal laser endomicroscopy.* Front Oncol. 2021;11:668273. doi:10.3389/fonc.2021.668273 | Brain, ex vivo; 13,972 images from 25 patients | GBM / metastasis / meningioma | ResNet | 2-class accuracy 90.9% (AUC 0.92); 3-class 85.8% | Y (PubMed) |
| [LiY2018] | Li Y, Charalampaki P, Liu Y, Yang GZ, Giannarou S. *Context aware decision support in neurosurgical oncology based on an efficient classification of endomicroscopic data.* Int J Comput Assist Radiol Surg. 2018;13(8):1187-1199. doi:10.1007/s11548-018-1806-7 | Brain pCLE video | GBM vs. meningioma | CNN + long-range temporal recursion | video accuracy 99.49% | Y (PubMed) |
| [Serbanescu2025] | Șerbănescu MS, Streba L, Demetrian AD, et al. *Transfer learning-based integration of dual imaging modalities for enhanced classification accuracy in confocal laser endomicroscopy of lung cancer.* Cancers (Basel). 2025;17(4):611. doi:10.3390/cancers17040611 | Lung pCLE; 40 patients; 800 benign + 800 malignant images | benign vs. malignant | AlexNet / GoogLeNet / ResNet, dual transfer learning (histology → pCLE) | AlexNet accuracy 94.97%, AUC 0.98 | Y (PubMed) |
| [Xu2026] | Xu Y, Luo R, Tang J, et al. *Artificial intelligence-assisted confocal laser endomicroscopy for predicting invasion depth of superficial esophageal mucosal lesions: a cohort study.* Clin Transl Gastroenterol. 2026;17(7):e01044. doi:10.14309/ctg.0000000000001044 | Esophagus pCLE; 1,234 images | LGIN vs. HGIN-SM1 | best of 6 deep-learning models | sensitivity 97.4%, specificity 92.6%, accuracy 95.3% | Y (PubMed) |

Observations for related work:
- Most CLE deep-learning work uses off-the-shelf heavy backbones (Inception-ResNet-V2, EfficientNetV2-S, ResNet, AlexNet) with transfer learning.
- No CLE paper found uses a mobile backbone (MobileNetV3/V4, EfficientViT, StarNet, LSNet) or an explicit gland-structure prior.
- Patient-level splitting is not universal. CCADS splits by patient; Udristoiu 2021 does not.
- Many papers use 2 classes. Gastric multi-class (5-class CCADS, 6-class MixSiam) is rarer.

---

## 3. Method references

### 3.1 Base backbone and efficient token mixers

- **[Howard2019-MobileNetV3]** Howard A, Sandler M, Chu G, Chen LC, Chen B, Tan M, Wang W, Zhu Y, Pang R, Vasudevan V, Le QV, Adam H. *Searching for MobileNetV3.* ICCV 2019, pp. 1314-1324. https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html. **Verified: Y (CVF page + PDF).**
  - Mechanism we cite: NAS-designed inverted-residual blocks with SE and h-swish.
  - Paper figures: V3-Large 1.0 has 75.2% top-1, 219M MAdds, 5.4M parameters.
  - torchvision weights (docs page verified):

| Weights | acc@1 | Params | GFLOPS | URL |
|---|---|---|---|---|
| `IMAGENET1K_V1` | 74.042 | 5,483,032 | 0.22 | https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth |
| `IMAGENET1K_V2` | 75.274 | 5,483,032 | 0.22 | https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth |

- **[Wang2025-LSNet]** Wang A, Chen H, Lin Z, Han J, Ding G. *LSNet: See Large, Focus Small.* CVPR 2025, pp. 9718-9729. arXiv:2503.23135. https://openaccess.thecvf.com/content/CVPR2025/html/Wang_LSNet_See_Large_Focus_Small_CVPR_2025_paper.html. **Verified: Y (CVF page + PDF).**
  - LS convolution has two steps:
    - **Large-Kernel Perception (LKP):** PW (to C/2) → DW K_L×K_L → PW. This generates **per-pixel** context-adaptive weights W ∈ R^{H×W×D}.
    - **Small-Kernel Aggregation (SKA):** a grouped dynamic convolution. Channels are split into G groups (C/G = 8 channels share one kernel). Each pixel's weights are reshaped to w* ∈ R^{G×K_S×K_S} and aggregate its K_S×K_S neighbourhood: y_ic = w*_ig ⊛ N_KS(x_ic).
  - Defaults: K_L = 7, K_S = 3. The code uses `LKP(dim, lks=7, sks=3, groups=8)`, and the LS block adds BatchNorm and a residual.
  - Ablations on LSNet-T (shortened schedule):
    - Removing LKP costs 1.1 points top-1; removing SKA costs 1.5.
    - Against other dynamic convolutions at similar FLOPs (Table 6): LS conv 71.6 vs. CondConv 69.8, DY-Conv 70.0, Involution 70.3, DCD 69.8, CoT 71.1, ODConv 70.0. This is a ready-made justification for (A).
  - **Implementation note:** the official `model/ska.py` imports `triton` and wraps Triton JIT kernels in an autograd Function (CUDA only). A pure-PyTorch re-implementation (e.g., `unfold` + einsum) is needed for portability.
- **[Ma2024-StarNet]** Ma X, Dai X, Bai Y, Wang Y, Fu Y. *Rewrite the Stars.* CVPR 2024, pp. 5694-5703. arXiv:2403.19967. https://openaccess.thecvf.com/content/CVPR2024/html/Ma_Rewrite_the_Stars_CVPR_2024_paper.html. **Verified: Y (CVF page + PDF).**
  - The "star operation" is element-wise multiplication of two linear branches, (W1ᵀX+B1) ∗ (W2ᵀX+B2). It implicitly maps a d-dimensional input into a (d+2)(d+1)/2 ≈ d²/2-dimensional nonlinear feature space, like a polynomial kernel, and stacking layers raises the implicit dimension exponentially.
  - Relevance: this is the same-input special case of Hadamard low-rank bilinear pooling (MLB). Our (B) uses two different scales as inputs.
- **[Yu2024-InceptionNeXt]** Yu W, Zhou P, Yan S, Wang X. *InceptionNeXt: When Inception Meets ConvNeXt.* CVPR 2024, pp. 5672-5683. arXiv:2303.16900. https://openaccess.thecvf.com/content/CVPR2024/html/Yu_InceptionNeXt_When_Inception_Meets_ConvNeXt_CVPR_2024_paper.html. **Verified: Y (CVF page + PDF).**
  - Inception depthwise convolution (IDC) splits channels into four parts: a 3×3 DW branch, a 1×k_b DW branch, a k_b×1 DW branch (k_b = 11), and an identity branch. The default branch ratio is 1/8 of channels per conv branch.
  - Caution (our inference): the band kernels are **axis-aligned**. CLE fields are rotation-invariant textures, so this orientation prior may not fit.
- **[Chen2023-FasterNet]** Chen J, Kao SH, He H, Zhuo W, Wen S, Lee CH, Chan SHG. *Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks.* CVPR 2023, pp. 12021-12031. arXiv:2303.03667. https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Run_Dont_Walk_Chasing_Higher_FLOPS_for_Faster_Neural_Networks_CVPR_2023_paper.html. **Verified: Y (CVF page + PDF).**
  - Partial convolution (PConv) applies a regular conv to only c_p = c/4 channels and leaves the rest untouched. That costs 1/16 of a regular conv's FLOPs and about 1/4 of its memory access, and it is followed by a PWConv.

### 3.2 Bilinear and multiplicative feature interaction (for innovation B)

- **[Lin2015-BCNN]** Lin TY, RoyChowdhury A, Maji S. *Bilinear CNN Models for Fine-Grained Visual Recognition.* ICCV 2015, pp. 1449-1457. https://openaccess.thecvf.com/content_iccv_2015/html/Lin_Bilinear_CNN_Models_ICCV_2015_paper.html. **Verified: Y (CVF).**
  - Takes the outer product of two CNN feature maps at each location and sum-pools it, giving orderless second-order statistics.
  - The abstract notes it "generalizes various orderless texture descriptors such as the Fisher vector, VLAD and O2P". Relevant because CLE images are texture-like.
- **[Kim2017-MLB]** Kim JH, On KW, Lim W, Kim J, Ha JW, Zhang BT. *Hadamard Product for Low-rank Bilinear Pooling.* ICLR 2017. arXiv:1610.04325; OpenReview id r1rhWnZkg. **Verified: Y** (official code repo github.com/jnhwkim/MulLowBiVQA cites "The 5th International Conference on Learning Representations", 2017; arXiv and OpenReview themselves were unreachable from here).
  - Low-rank bilinear pooling f = Pᵀ(σ(Uᵀx) ∘ σ(Vᵀy)) + b. A full bilinear map is replaced by projection to a shared rank-d space plus a Hadamard product.
  - This is the formal identity behind our cross-scale star head (x = fine-scale feature, y = coarse-scale feature).
- **[Yu2017-MFB]** Yu Z, Yu J, Fan J, Tao D. *Multi-modal Factorized Bilinear Pooling with Co-Attention Learning for Visual Question Answering.* ICCV 2017, pp. 1821-1830. arXiv:1708.01471. https://openaccess.thecvf.com/content_iccv_2017/html/Yu_Multi-Modal_Factorized_Bilinear_ICCV_2017_paper.html. **Verified: Y (CVF page + PDF page numbers).**
  - Both inputs are projected to k·o dimensions, multiplied with a Hadamard product and sum-pooled over windows of size k (a rank-k factorisation per output). This is followed by power and L2 normalisation, which is useful for stabilising multiplicative heads.
- **[Yu2018-HBP]** Yu C, Zhao X, Zheng Q, Zhang P, You X. *Hierarchical Bilinear Pooling for Fine-Grained Visual Recognition.* ECCV 2018. arXiv:1807.09915. https://openaccess.thecvf.com/content_ECCV_2018/html/Chaojian_Yu_Hierarchical_Bilinear_Pooling_ECCV_2018_paper.html. **Verified: Y (CVF).**
  - **Cross-layer** bilinear pooling: a Hadamard product of projected features from different layers, with several such pairs integrated hierarchically.
  - **The closest prior to (B).** It must be cited and differentiated: our version is a lightweight evidence head over gland-scale and cell-scale features in a mobile CNN, motivated by CLE criteria.

### 3.3 MatchED, boundary/structure supervision and distance-based losses (for innovation C)

- **[Cetinkaya2026-MatchED]** Cetinkaya B, Kalkan S, Akbas E. *MatchED: Crisp Edge Detection Using End-to-End, Matching-based Supervision.* CVPR 2026, pp. 42093-42103. arXiv:2602.20689. Project/code: https://cvpr26-matched.github.io. CVF: https://openaccess.thecvf.com/content/CVPR2026/html/Cetinkaya_MatchED_Crisp_Edge_Detection_Using_End-to-End_Matching-based_Supervision_CVPR_2026_paper.html. **Verified: Y (CVF page + full PDF read).**
  - **What it is:** a lightweight (~21K parameters; 5 Conv-ReLU-Norm blocks + Conv + sigmoid), plug-and-play module appended to *any edge detector* (they test PiDiNet, RankED, DiffusionEdge, SAUGE). It produces one-pixel-wide ("crisp") edges without NMS or thinning.
  - **Loss mechanism:**
    1. At each training iteration, build a cost matrix between predicted-edge pixels p_c and ground-truth edge pixels p_g: C = d_Manhattan(p_c, p_g) − α·E_c(p_c). An entry is valid only if E_c(p_c) ≥ τ_c, G(p_g) = 1 and d < τ_d; otherwise it is ∞.
    2. Solve the **optimal one-to-one assignment** with linear sum assignment to get a matched target Ĝ. Unmatched ground-truth edge pixels are kept as positives.
    3. Train the module with **BCE(E_c, Ĝ)**. The total loss is β·L_MatchED + L_f.
  - τ_d is set to 8 / 4 / 11 / 11 px for NYUD-v2 / BSDS / Multi-cue / BIPED. The base detector is trained alone for N/2 epochs, then jointly with MatchED.
  - Results: Average Crispness up to 2–4× higher; ODS up to +20–35% under the crispness-emphasised evaluation (CEval).
  - **Important:** MatchED does **not** define a chamfer or "soft-chamfer" loss. Its tolerance-aware, distance-thresholded matching is the conceptual inspiration. A *soft-chamfer* (differentiable bidirectional nearest-distance) gland-boundary loss is our own relaxation and should be cited to the chamfer and boundary-loss works below.
- **[Fan2017-Chamfer]** Fan H, Su H, Guibas LJ. *A Point Set Generation Network for 3D Object Reconstruction from a Single Image.* CVPR 2017. arXiv:1612.00603. https://openaccess.thecvf.com/content_cvpr_2017/html/Fan_A_Point_Set_CVPR_2017_paper.html. **Verified: Y (CVF).** Standard deep-learning citation for the Chamfer distance used as a training loss.
- **[Kervadec2021]** Kervadec H, Bouchtiba J, Desrosiers C, Granger E, Dolz J, Ben Ayed I. *Boundary loss for highly unbalanced segmentation.* Med Image Anal. 2021;67:101851. doi:10.1016/j.media.2020.101851. PMID 33080507. **Verified: Y (PubMed).** Distance-transform-based boundary loss in medical imaging.
- **[Karimi2020]** Karimi D, Salcudean SE. *Reducing the Hausdorff distance in medical image segmentation with convolutional neural networks.* IEEE Trans Med Imaging. 2020;39(2):499-513. doi:10.1109/TMI.2019.2930068. PMID 31329113. **Verified: Y (PubMed).** Differentiable distance-transform losses that approximate Hausdorff distance.
- **[Chen2016-DCAN]** Chen H, Qi X, Yu L, Heng PA. *DCAN: Deep Contour-Aware Networks for Accurate Gland Segmentation.* CVPR 2016. https://openaccess.thecvf.com/content_cvpr_2016/html/Chen_DCAN_Deep_Contour-Aware_CVPR_2016_paper.html. **Verified: Y (CVF).** Journal version: Chen H, Qi X, Yu L, Dou Q, Qin J, Heng PA. Med Image Anal. 2017;36:135-146. doi:10.1016/j.media.2016.11.004. PMID 27898306 (Verified: Y).
  - Multi-task gland object plus **gland contour** prediction with auxiliary supervision. The abstract states "The morphology of glands has been used routinely by pathologists to assess the malignancy degree of adenocarcinomas."
  - The precedent for supervising gland boundaries.
- **[Awan2017]** Awan R, Sirinukunwattana K, Epstein D, Jefferyes S, Qidwai U, Aftab Z, Mujeeb I, Snead D, Rajpoot N. *Glandular morphometrics for objective grading of colorectal adenocarcinoma histology images.* Sci Rep. 2017;7(1):16852. doi:10.1038/s41598-017-16516-w. PMID 29203775. **Verified: Y (PubMed).**
  - Gland boundaries are detected with a CNN, then a gland shape metric (BAM) is computed and fed to an SVM. Accuracy was 97% for 2 classes and 91% for 3 classes (normal / low-grade / high-grade).
  - Direct evidence (in histology) that **gland-shape irregularity carries the benign/malignant signal**.
- **[Zhou2019-Collab]** Zhou Y, He X, Huang L, Liu L, Zhu F, Cui S, Shao L. *Collaborative Learning of Semi-Supervised Segmentation and Classification for Medical Images.* CVPR 2019. https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_Collaborative_Learning_of_Semi-Supervised_Segmentation_and_Classification_for_Medical_Images_CVPR_2019_paper.html. **Verified: Y (CVF).** Joint lesion segmentation and disease grading, where pixel-level supervision on a small subset improves classification.
- **[Xie2020-MBDCNN]** Xie Y, Zhang J, Xia Y, Shen C. *A mutual bootstrapping model for automated skin lesion segmentation and classification.* IEEE Trans Med Imaging. 2020;39(7):2482-2493. doi:10.1109/TMI.2020.2972964. PMID 32070946. **Verified: Y (PubMed).** Segmentation and classification networks bootstrap each other. Classification AUC 93.8% on ISIC-2017 and 97.7% on PH2.
- **[Geirhos2019]** Geirhos R, Rubisch P, Michaelis C, Bethge M, Wichmann FA, Brendel W. *ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness.* ICLR 2019 (oral). OpenReview id Bygh9j09KX. **Verified: Y (official GitHub rgeirhos/texture-vs-shape).** Rationale for adding structure/boundary supervision to an ImageNet-pretrained, texture-biased CNN.

### 3.4 Dynamic / content-adaptive convolution (context for innovation A)

- **[Li2021-Involution]** Li D, Hu J, Wang C, Li X, She Q, Zhu L, Zhang T, Chen Q. *Involution: Inverting the Inherence of Convolution for Visual Recognition.* CVPR 2021, pp. 12321-12330. arXiv:2103.06255. **Verified: Y (CVF).** Kernels are **spatial-specific and channel-agnostic**, generated from the feature at each location. This inverts the spatial-agnostic, channel-specific design of convolution.
- **[Chen2020-DyConv]** Chen Y, Dai X, Liu M, Chen D, Yuan L, Liu Z. *Dynamic Convolution: Attention over Convolution Kernels.* CVPR 2020. arXiv:1912.03458. **Verified: Y (CVF).** Aggregates K parallel kernels per layer with **input-dependent attention**. The kernel adapts per sample but is shared across space, and the paper targets light-weight CNNs, including MobileNetV3.
- **[Zhou2021-DDF]** Zhou J, Jampani V, Pi Z, Liu Q, Yang MH. *Decoupled Dynamic Filter Networks.* CVPR 2021, pp. 6647-6656. arXiv:2104.14107. **Verified: Y (CVF).** Decouples a depthwise dynamic filter into a spatial dynamic filter and a channel dynamic filter, keeping cost at depthwise-conv level.

Positioning for (A): LSNet's SKA is per-pixel (like involution/DDF) but its kernel is predicted from a large 7×7 context (LKP). That is "see large, focus small", which matches judging gland regularity from context while aggregating locally.

---

## 4. Lightweight backbones with official ImageNet-1K weights (comparison-table candidates)

All values below are ImageNet-1K at 224 px unless noted.
- **Params/MACs** are the official paper or README numbers. Where the timm model card differs, it is noted.
- **Top-1** is the paper value; "rel." is the official released-checkpoint README value when it differs.
- "dist." means trained with knowledge distillation.

| Model | Params (M) | MACs (G) | Top-1 (%) | Official weights (verified URL) | timm name (HF `timm/…`) | Notes |
|---|---|---|---|---|---|---|
| MobileNetV3-L (ours) | 5.48 | 0.22 | 75.274 (V2) / 74.042 (V1) | download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth | timm equivalents exist (`mobilenetv3_large_100.ra_in1k`, `.ra4_e3600_r224_in1k`) but are *different* weights from torchvision | torchvision; pure PyTorch |
| **LSNet-T** | 11.4 | 0.3 | 74.9 (76.1 dist.) | https://huggingface.co/jameslahm/lsnet/blob/main/lsnet_t.pth (and `lsnet_t_distill.pth`) | not in timm; `timm.create_model('hf_hub:jameslahm/lsnet_t')` with repo code | **SKA needs Triton/CUDA**; MIT |
| **LSNet-S** | 16.1 | 0.5 | 77.8 (79.0 dist.) | …/jameslahm/lsnet/blob/main/lsnet_s.pth | `hf_hub:jameslahm/lsnet_s` | as above |
| (LSNet-B) | 23.2 | 1.3 | 80.3 (81.6 dist.) | …/lsnet_b.pth | `hf_hub:jameslahm/lsnet_b` | above budget |
| **StarNet-S1** | 2.9 | 0.425 | 73.5 (rel. 73.6) | https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar | `starnet_s1.in1k` | pure PyTorch; Apache-2.0 |
| **StarNet-S2** | 3.7 | 0.547 | 74.8 (rel. 74.7) | …/checkpoints_v1/starnet_s2.pth.tar | `starnet_s2.in1k` | |
| **StarNet-S3** | 5.8 | 0.757 | 77.3 (rel. 77.4) | …/checkpoints_v1/starnet_s3.pth.tar | `starnet_s3.in1k` | |
| **StarNet-S4** | 7.5 | 1.075 | 78.4 (rel. 78.8) | …/checkpoints_v1/starnet_s4.pth.tar | `starnet_s4.in1k` | |
| **MobileNetV4-Conv-S** | 3.77 | 0.2 | **timm weights:** 74.616 @256 test (e2400); 74.292 (e1200). Paper value UNVERIFIED | https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k | `mobilenetv4_conv_small.e2400_r224_in1k` | Weights trained by R. Wightman with timm. The card says official TF weights were "unreleased". Train 224, test 256 |
| **MobileNetV4-Conv-M** | 9.72 | 0.8 @224 (1.1 @256) | timm: 79.094 @224 / 79.808 @256 (e500_r224); 79.928 @256 / 80.858 @320 (e500_r256) | https://huggingface.co/timm/mobilenetv4_conv_medium.e500_r224_in1k | `mobilenetv4_conv_medium.e500_r224_in1k` | same caveat |
| EfficientViT-M0 / M1 / M2 | 2.3 / 3.0 / 4.2 | 0.079 / 0.167 / 0.201 | 63.2 / 68.4 / 70.8 | https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/efficientvit_m{0,1,2}.pth | `efficientvit_m{0,1,2}.r224_in1k` | Microsoft (MSRA) CVPR'23; MIT; pure PyTorch |
| **EfficientViT-M3 / M4 / M5** | 6.9 / 8.8 / 12.4 | 0.263 / 0.299 / 0.522 | 73.4 / 74.3 / 77.1 | …/v1.0/efficientvit_m{3,4,5}.pth | `efficientvit_m{3,4,5}.r224_in1k` | M4/M5 sit at our budget |
| RepViT-M0.9 (have) | 5.1 | 0.8 | 78.7 / 79.1 dist. (300/450 ep); 77.4 without distillation | (already available) | `repvit_m0_9.dist_300e_in1k` / `.dist_450e_in1k` | timm card: 5.5M / 0.8 GMACs |
| RepViT-M1.0 / M1.1 | 6.8 / 8.2 | 1.1 / 1.3 | 80.0/80.3; 80.7/81.2 dist. | — | `repvit_m1_0.dist_450e_in1k`, `repvit_m1_1.dist_450e_in1k` | |
| **EfficientFormerV2-S0** | 3.5 | 0.40 | 75.7 / 76.2 dist. (300/450 ep) | Google Drive link in snap-research/EfficientFormer README (S0 checkpoint: https://drive.google.com/file/d/1PXb7b9pv9ZB4cfkRkYEdwgWuVwvEiazq/view) | `efficientformerv2_s0.snap_dist_in1k` | distilled; use timm for convenience |
| EfficientFormerV2-S1 / S2 | 6.1 / 12.6 | 0.65 / 1.25 | 79.0/79.7; 81.6/82.0 dist. | same README | `efficientformerv2_s{1,2}.snap_dist_in1k` | |
| **FastViT-T8** | 3.6 | 0.7 (256 px) | 75.6 (76.7 dist.); rel. 76.2 (77.2 dist.) | https://docs-assets.developer.apple.com/ml-research/models/fastvit/image_classification_models/fastvit_t8_reparam.pth.tar (unfused also given) | `fastvit_t8.apple_in1k` / `.apple_dist_in1k` | 256 px input; Apple sample-code licence ("other"); reparameterisable |
| FastViT-T12 / S12 / SA12 | 6.8 / 8.8 / 10.9 | 1.4 / 1.8 / 1.9 (256 px) | 79.1 / 79.8 / 80.6 (rel. 79.3 / 79.9 / 80.9) | same base URL, `fastvit_{t12,s12,sa12}_reparam.pth.tar` | `fastvit_{t12,s12,sa12}.apple_in1k` | |
| **GhostNetV2-1.0×** | 6.2 (timm card) | 0.167 | 75.3 (official README) | https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/GhostNetV2/ck_ghostnetv2_10.pth.tar | `ghostnetv2_100.in1k` | NeurIPS 2022; pure PyTorch |
| GhostNetV2-1.3× / 1.6× | 9.0 / 12.4 (timm card) | ~0.3 / ~0.4 (timm card) | **UNVERIFIED** (paper table not reachable) | …/GhostNetV2/ck_ghostnetv2_13.pth.tar, …/ck_ghostnetv2_16.pth.tar | `ghostnetv2_130.in1k`, `ghostnetv2_160.in1k` | |
| GhostNetV3-1.0× | 8.1 / 0.4 G per timm card (likely training-time, un-reparameterised graph) | — | **UNVERIFIED** for 1.0×. Verified for 1.3×: 79.1% at 269M FLOPs | timm source comment: https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/GhostNetV3/ghostnetv3-1.0.pth.tar | `ghostnetv3_100.in1k` | arXiv 2404.11202 (not peer reviewed); uses reparameterisation (timm `convert_to_deploy()`) |
| InceptionNeXt-A (atto) | 4.2 | 0.51 | 75.3 | https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_atto.pth | `inception_next_atto.sail_in1k` | IDC reference model |
| FasterNet-T0 / T1 / T2 | 3.9 / 7.6 / 15.0 | 0.34 / 0.85 / 1.91 | 71.9 / 76.2 / 78.9 | https://github.com/JierunChen/FasterNet/releases/download/v1.0/fasternet_t0-epoch.281-val_acc1.71.9180.pth (also `fasternet_t1-epoch.291-val_acc1.76.2180.pth`, `fasternet_t2-epoch.289-val_acc1.78.8860.pth`) | `fasternet_t0.in1k` / `t1` / `t2` | PConv reference model |

Sources:
- Paper tables verified from CVF PDFs: LSNet, StarNet, EfficientViT, RepViT, FastViT, EfficientFormerV2, InceptionNeXt, FasterNet, MobileNetV3.
- Release URLs verified from the official READMEs: LSNet HF repo (via HF API file list), StarNet `imagenet/` README, Cream/EfficientViT classification README, ml-fastvit README, EfficientFormer README, GhostNetV2 README, InceptionNeXt README, FasterNet README.
- timm names verified with the HF API (`author=timm`). MobileNetV4 timm top-1 comes from the comparison table in the timm model card.

Comparability caveats for the paper's table:
1. Several published numbers use distillation (RepViT, EfficientFormerV2, `*_dist` FastViT, LSNet `*`). Report the non-distilled number where one exists, or label it.
2. FastViT uses 256 px input, and MNv4 timm weights are evaluated at 256 px.
3. Reparameterisable models (FastViT, RepViT, GhostNetV3) have different parameter counts before and after fusion.
4. **Suggested budget-matched set (≈0.2–0.6 GMACs):**
   - MobileNetV3-L (0.22)
   - MNv4-Conv-S (0.2)
   - GhostNetV2-1.0 (0.17)
   - EfficientViT-M4 (0.30) and M5 (0.52)
   - LSNet-T (0.3)
   - FasterNet-T0 (0.34)
   - EfficientFormerV2-S0 (0.40)
   - StarNet-S1 (0.43) and S2 (0.55)
   - InceptionNeXt-A (0.51)
   - RepViT-M0.9 (0.8) and FastViT-T8 (0.7 @256) sit slightly above this range.
5. All candidates except LSNet load in pure PyTorch through timm.

Correct citations for backbones not cited in Section 3 (verified from CVF unless noted):
- Liu X, Peng H, Zheng N, Yang Y, Hu H, Yuan Y. *EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention.* CVPR 2023, pp. 14420-14430. arXiv:2305.07027. Y.
- Wang A, Chen H, Lin Z, Han J, Ding G. *RepViT: Revisiting Mobile CNN From ViT Perspective.* CVPR 2024, pp. 15909-15920. arXiv:2307.09283. Y.
- Li Y, Hu J, Wen Y, Evangelidis G, Salahi K, Wang Y, Tulyakov S, Ren J. *Rethinking Vision Transformers for MobileNet Size and Speed* (EfficientFormerV2). ICCV 2023, pp. 16889-16900. arXiv:2212.08059. Y.
- Vasu PKA, Gabriel J, Zhu J, Tuzel O, Ranjan A. *FastViT: A Fast Hybrid Vision Transformer Using Structural Reparameterization.* ICCV 2023, pp. 5785-5795. arXiv:2303.14189. Y.
- Han K, Wang Y, Tian Q, Guo J, Xu C, Xu C. *GhostNet: More Features From Cheap Operations.* CVPR 2020. arXiv:1911.11907. Y (CVF).
- Tang Y, Han K, Guo J, Xu C, Xu C, Wang Y. *GhostNetV2: Enhance Cheap Operation with Long-Range Attention.* NeurIPS 2022 (Spotlight). arXiv:2211.12905. Existence Y (official README says "NeurIPS 2022 Spotlight"). Volume/pages (35:9969–9982) come only from the timm card: UNVERIFIED.
- Liu Z, Hao Z, Han K, Tang Y, Wang Y. *GhostNetV3: Exploring the Training Strategies for Compact Models.* arXiv:2404.11202, 2024. Y (alphaXiv abstract page). The timm card's BibTeX author list is **wrong**.
- Qin D, Leichner C, Delakis M, Fornoni M, Luo S, Yang F, Wang W, Banbury C, Ye C, Akin B, et al. *MobileNetV4: Universal Models for the Mobile Ecosystem.* ECCV 2024. arXiv:2404.10518. Existence via the timm card and search; **the ECVA page was unreachable, so the venue and paper numbers are UNVERIFIED here**.
- The StarNet timm-card BibTeX author list is **wrong**. Use the CVF list (Ma, Dai, Bai, Wang, Fu).

---

## 5. Items not verified or excluded

- Zheng X, et al. 2024, J King Saud Univ Comput Inf Sci, "pCLE diagnosis area identification model (pCLEDAM)" for gastric endoscopic video (ScienceDirect S1319157824002970). The page returned 403 and the task is peripheral (site localisation, not image classification). **UNVERIFIED; excluded.**
- Liu YS, et al. 2025, review of AI-assisted CLE in GI diseases (wjgnet.com 2644-3260 v6 i1 106928). The page returned 403. **UNVERIFIED; excluded.**
- MobileNetV4 paper-reported top-1 (Conv-S, Conv-M); GhostNetV2 1.3×/1.6× top-1; GhostNetV3 1.0× top-1. **UNVERIFIED** (Section 4).
- Zhang 2008 verbatim definition and subtype names for type G. **UNVERIFIED** beyond the [Han2023] paraphrase.
- The original Mainz table is verified only through the secondary source [Ussui2012] (Kiesslich 2004 full text paywalled).
