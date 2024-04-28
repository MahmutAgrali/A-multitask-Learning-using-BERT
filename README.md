## Project Structure Overview / Proje Yapısı Genel Bakışı

### Directories / Dizinler
- **predictions**: Contains prediction results / Tahmin sonuçlarını içerir.
- **smart_pytorch**: Directory borrowed from GitHub / GitHub'dan ödünç alınan dizin.

### Files / Dosyalar
#### Scripts Modified for Project Scope / Projeye Kapsamına Göre Değiştirilen Betikler:
- **bert.py**: Implements the BERT model for this project / Bu projede BERT modelini uygular.
- **multitask_classifier.py**: Script for multitask learning, focused on sentiment classification / Çoklu görev öğrenimi için betik, duygu sınıflandırmasına odaklanıyor.
- **multitask_classifier_with_all_tasks.py**: Enhanced multitask learning script, yielding optimal results / En iyi sonuçları elde etmek için geliştirilmiş çoklu görev öğrenimi betiği.
- **classifier.py**: Script for training classification tasks / Sınıflandırma görevlerini eğitmek için betik.
- **optimizer.py**: Optimizer script / Optimizasyon betiği.
- **evaluation.py**: Script for evaluation / Değerlendirme için betik.

#### Scripts Borrowed from GitHub / GitHub'dan Ödünç Alınan Betikler:
- **triplet_loss.py**: Script for triplet loss, borrowed from GitHub / GitHub'dan ödünç alınan üçlü kayıp için betik.
- **pcgrad.py**: Implementation of PCGrad, sourced from GitHub ([GitHub Repository](https://github.com/WeiChengTseng/Pytorch-PCGrad/blob/master/pcgrad.py)) / PCGrad'ın uygulanması, GitHub'dan alınmıştır.

#### Additional Files / Ek Dosyalar:
- **base_bert.py**: Script for the base BERT model / Temel BERT modeli için betik.
- **config.py**: Configuration script for training and evaluation / Eğitim ve değerlendirme yapılandırması için betik.
- **datatests.py**: Script necessary for dataset handling / Veri kümesi işleme için gerekli betik.
- **optimizer_test.py**: Test script for optimizer functionality / Optimizasyon işlevselliği için test betiği.
- **prepare_submit.py**: Script used for submission purposes / Gönderim amaçlı kullanılan betik.
- **sanity_check.py**: Test script for validating classifier.py functionality / classifier.py işlevselliğini doğrulamak için test betiği.
- **tokenizer.py**: Tokenizer script for dataset processing / Veri kümesi işleme için belirteçleyici betik.
- **utils.py**: Script containing helpful utility functions / Faydalı yardımcı işlevleri içeren betik.

This directory structure and file organization facilitate clarity and modularity within the BERT-default-final-project. Each script and directory serves a specific purpose, contributing to the project's overall functionality and effectiveness. / Bu dizin yapısı ve dosya düzeni, BERT-default-final-project içinde netlik ve modülerlik sağlar. Her betik ve dizin belirli bir amaca hizmet eder, projenin genel işlevselliğine ve etkinliğine katkıda bulunur.
