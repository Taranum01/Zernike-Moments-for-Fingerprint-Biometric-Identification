# 🔍 Zernike Moments for Fingerprint Biometric Identification  

This project explores the use of **Zernike Moment (ZM) vectors** for biometric fingerprint recognition using the **SOCOFing dataset** (55,000+ fingerprint images from 600 subjects). Zernike Moments are chosen due to their **orthogonality, rotation invariance, and robustness to noise**, making them highly effective for feature extraction in biometric systems.  

## 📌 Key Highlights  
- **Feature Extraction:** Orthogonal radial Zernike Moments computed for fingerprint images.  
- **Similarity Metrics:** Comparison between **Euclidean Distance** and **Cosine Similarity**.  
- **Performance Results:**  
  - ✅ 95.28% accuracy with Cosine similarity  
  - ✅ 92.11% accuracy with Euclidean distance  
  - 👍 Thumb fingerprints yielded the highest reliability (~97% accuracy)  
  - 👎 Little finger showed slightly lower performance (~94%)  
- **Intra-class vs. Inter-class Analysis:** Verified that fingerprints of the same person cluster closely, while different individuals’ prints remain well-separated.  
- **Error Analysis:** Identified challenging cases of misclassification, guiding future improvements.  

## 📊 Conclusion  
The results confirm that **Zernike Moments provide a discriminative and reliable feature representation** for fingerprint-based biometric identification. Cosine similarity consistently outperformed Euclidean distance, highlighting the importance of directional vector properties in biometric matching.  
