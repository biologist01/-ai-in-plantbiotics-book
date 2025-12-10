import os
import asyncio
import sys
from groq import AsyncGroq
from pathlib import Path
from dotenv import load_dotenv

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

# Documentation structure for AI Revolution in Plant Biotechnology
DOCS_STRUCTURE = {
    "module-1": [
        {
            "filename": "classification-models.md",
            "title": "Classification Models for Plant Analysis",
            "sidebar_position": 3,
            "topic": "Build classification models for plant species identification, disease detection, and health status prediction. Cover decision trees, random forests, SVM, and gradient boosting with practical agricultural examples.",
            "key_points": [
                "Binary and multi-class classification in agriculture",
                "Decision trees and random forests for plant classification",
                "Support Vector Machines for disease detection",
                "Gradient boosting (XGBoost, LightGBM) for high accuracy",
                "Model evaluation metrics (accuracy, precision, recall, F1)",
                "Handling imbalanced datasets (healthy vs diseased)",
                "Feature importance and interpretability",
                "Practical project: Plant disease classifier with 95%+ accuracy"
            ]
        },
        {
            "filename": "regression-models.md",
            "title": "Regression Models for Yield Prediction",
            "sidebar_position": 4,
            "topic": "Learn regression techniques to predict crop yields, plant growth rates, and harvest timing. Cover linear regression, polynomial regression, ensemble methods with real agricultural datasets.",
            "key_points": [
                "Linear and polynomial regression fundamentals",
                "Multiple regression with agricultural features",
                "Regularization (Lasso, Ridge) to prevent overfitting",
                "Ensemble regression methods (Random Forest, Gradient Boosting)",
                "Time-to-harvest prediction models",
                "Soil nutrient impact on yield prediction",
                "Model evaluation (RMSE, MAE, R² score)",
                "Practical project: Wheat yield prediction system"
            ]
        },
        {
            "filename": "time-series.md",
            "title": "Time-Series Analysis for Crop Monitoring",
            "sidebar_position": 5,
            "topic": "Master time-series forecasting for plant growth patterns, seasonal trends, and environmental monitoring. Cover ARIMA, Prophet, LSTM networks for sequential agricultural data.",
            "key_points": [
                "Time-series data in agriculture (growth, weather, sensors)",
                "Stationarity and differencing",
                "ARIMA models for crop growth forecasting",
                "Facebook Prophet for seasonal patterns",
                "LSTM neural networks for complex sequences",
                "Multi-variate time-series analysis",
                "Forecasting soil moisture and irrigation needs",
                "Practical project: 30-day growth prediction system"
            ]
        },
        {
            "filename": "ml-project.md",
            "title": "Mini-Project: Disease Prediction System",
            "sidebar_position": 6,
            "topic": "Build a complete end-to-end machine learning system for early disease detection using sensor data and environmental factors. Integrate data collection, preprocessing, model training, and deployment.",
            "key_points": [
                "Project overview: Early disease warning system",
                "Dataset collection and preparation",
                "Feature engineering from sensor data",
                "Multi-model comparison and selection",
                "Hyperparameter tuning with GridSearch/RandomSearch",
                "Model deployment strategies",
                "Building a simple prediction API with FastAPI",
                "Real-time monitoring dashboard"
            ]
        }
    ],
    "module-2": [
        {
            "filename": "cv-intro.md",
            "title": "Introduction to Computer Vision in Agriculture",
            "sidebar_position": 1,
            "topic": "Explore how computer vision revolutionizes agriculture through automated plant monitoring, disease detection, and phenotyping. Learn image fundamentals and agricultural imaging systems.",
            "key_points": [
                "Computer vision applications in agriculture",
                "Image formation and digital representation",
                "Color spaces (RGB, HSV, LAB) for plant analysis",
                "Image acquisition systems (cameras, drones, satellites)",
                "Lighting conditions and image quality",
                "Plant imaging best practices",
                "Common agricultural image datasets (PlantVillage, etc.)",
                "Introduction to OpenCV for plant image processing"
            ]
        },
        {
            "filename": "image-processing.md",
            "title": "Image Acquisition and Preprocessing",
            "sidebar_position": 2,
            "topic": "Learn essential image preprocessing techniques for plant analysis including filtering, segmentation, feature extraction, and background removal for robust plant phenotyping.",
            "key_points": [
                "Image enhancement and noise reduction",
                "Background removal for plant isolation",
                "Color-based segmentation for leaf detection",
                "Morphological operations (erosion, dilation)",
                "Edge detection for leaf boundaries",
                "Contour detection and analysis",
                "Feature extraction (color histograms, texture, shape)",
                "Practical project: Automated leaf segmentation"
            ]
        },
        {
            "filename": "deep-learning-cnn.md",
            "title": "Deep Learning for Plant Disease Detection",
            "sidebar_position": 3,
            "topic": "Master Convolutional Neural Networks (CNNs) for plant disease classification. Learn transfer learning with pre-trained models, data augmentation, and deployment strategies.",
            "key_points": [
                "CNN architecture fundamentals (convolution, pooling, FC layers)",
                "Building CNNs with TensorFlow/Keras and PyTorch",
                "Transfer learning with ResNet, VGG, EfficientNet",
                "Data augmentation for limited agricultural datasets",
                "Training strategies and regularization",
                "Multi-class disease classification",
                "Model interpretation with Grad-CAM",
                "Practical project: 20+ disease classifier with 98%+ accuracy"
            ]
        },
        {
            "filename": "object-detection.md",
            "title": "Object Detection for Fruits and Flowers",
            "sidebar_position": 4,
            "topic": "Implement object detection models for counting fruits, detecting flowers, and automated harvesting. Cover YOLO, Faster R-CNN, and custom detection pipelines.",
            "key_points": [
                "Object detection vs classification vs segmentation",
                "YOLO architecture and real-time detection",
                "Faster R-CNN for precise localization",
                "Training custom object detectors",
                "Annotation tools (LabelImg, CVAT) for agricultural data",
                "Fruit counting and yield estimation",
                "Maturity detection for harvest timing",
                "Practical project: Automated tomato detection and counting"
            ]
        },
        {
            "filename": "cv-project.md",
            "title": "Mini-Project: Automated Plant Phenotyping",
            "sidebar_position": 5,
            "topic": "Build a complete automated phenotyping system that measures plant height, leaf area, color analysis, and growth tracking from image sequences using computer vision.",
            "key_points": [
                "Project overview: High-throughput phenotyping system",
                "Multi-view image capture setup",
                "Plant segmentation and 3D reconstruction",
                "Automated measurement extraction (height, width, leaf count)",
                "Leaf area calculation using pixel analysis",
                "Color analysis for health assessment",
                "Time-lapse growth tracking",
                "Export data for ML analysis"
            ]
        }
    ],
    "module-3": [
        {
            "filename": "genomics-intro.md",
            "title": "Introduction to AI in Plant Genomics",
            "sidebar_position": 1,
            "topic": "Understand how AI accelerates plant genomics research, from DNA sequencing analysis to trait prediction. Learn genomic data formats and basic bioinformatics concepts.",
            "key_points": [
                "Plant genomics fundamentals (DNA, genes, traits)",
                "Genomic data formats (FASTA, FASTQ, VCF, GFF)",
                "High-throughput sequencing technologies",
                "AI applications in genomics (variant calling, annotation, prediction)",
                "Genotype to phenotype mapping",
                "GWAS (Genome-Wide Association Studies)",
                "Introduction to bioinformatics tools and databases",
                "Practical example: Analyzing Arabidopsis genome data"
            ]
        },
        {
            "filename": "sequence-analysis.md",
            "title": "Deep Learning for Genomic Sequences",
            "sidebar_position": 2,
            "topic": "Apply deep learning to genomic sequence analysis including promoter prediction, splice site detection, and gene finding using CNNs and RNNs on DNA sequences.",
            "key_points": [
                "Encoding DNA sequences for neural networks (one-hot, k-mer)",
                "CNNs for motif discovery in regulatory regions",
                "RNNs and LSTMs for sequence modeling",
                "Transformer models for long-range dependencies",
                "Promoter and enhancer prediction",
                "Splice site detection with deep learning",
                "Protein function prediction from sequences",
                "Practical project: Gene expression prediction from promoter sequences"
            ]
        },
        {
            "filename": "crispr-ai.md",
            "title": "CRISPR Target Prediction with AI",
            "sidebar_position": 3,
            "topic": "Learn how AI optimizes CRISPR gene editing by predicting guide RNA efficiency, off-target effects, and designing optimal editing strategies for crop improvement.",
            "key_points": [
                "CRISPR-Cas9 fundamentals and gene editing",
                "Guide RNA design challenges",
                "ML models for on-target activity prediction",
                "Off-target effect prediction and minimization",
                "Deep learning for sgRNA efficiency scoring",
                "Multiplexed editing strategy optimization",
                "AI-designed crops: examples and case studies",
                "Practical project: Design optimal CRISPR edits for drought resistance"
            ]
        },
        {
            "filename": "genomic-selection.md",
            "title": "Genomic Selection and Breeding",
            "sidebar_position": 4,
            "topic": "Master genomic selection techniques to accelerate crop breeding programs. Use ML to predict breeding values and design optimal crosses for desired traits.",
            "key_points": [
                "Traditional breeding vs genomic selection",
                "Genomic prediction models (GBLUP, rrBLUP, Bayesian)",
                "Deep learning for complex trait prediction",
                "Multi-trait and multi-environment models",
                "Training population design and optimization",
                "Breeding value prediction accuracy",
                "Optimal cross selection with genetic algorithms",
                "Practical project: Predict yield from genomic markers"
            ]
        },
        {
            "filename": "genomics-project.md",
            "title": "Mini-Project: Trait Prediction System",
            "sidebar_position": 5,
            "topic": "Build a genomic trait prediction system that predicts plant phenotypes from genotypic data using ensemble ML methods and validates predictions with real datasets.",
            "key_points": [
                "Project overview: Genomic-to-phenotypic prediction pipeline",
                "Dataset: Rice/Maize genomic and phenotypic data",
                "SNP filtering and quality control",
                "Feature engineering from genomic markers",
                "Ensemble methods for trait prediction",
                "Cross-validation with different environments",
                "Model interpretation: identifying causal variants",
                "Integration with breeding programs"
            ]
        }
    ],
    "module-4": [
        {
            "filename": "iot-intro.md",
            "title": "Introduction to IoT in Agriculture",
            "sidebar_position": 1,
            "topic": "Learn IoT fundamentals for smart agriculture including sensor networks, edge computing, and cloud integration for real-time farm monitoring and decision-making.",
            "key_points": [
                "IoT architecture for precision agriculture",
                "Sensor types (soil, weather, plant health)",
                "Microcontrollers (Arduino, Raspberry Pi, ESP32)",
                "Wireless communication (WiFi, LoRa, Bluetooth, Zigbee)",
                "Edge computing vs cloud processing",
                "Data protocols (MQTT, HTTP, CoAP)",
                "Power management and solar solutions",
                "Practical example: Build a soil moisture monitoring node"
            ]
        },
        {
            "filename": "sensor-networks.md",
            "title": "Sensor Networks and Data Collection",
            "sidebar_position": 2,
            "topic": "Design and deploy multi-sensor networks for comprehensive farm monitoring. Learn data aggregation, storage, and real-time streaming to ML pipelines.",
            "key_points": [
                "Sensor network topology design",
                "Multi-sensor data fusion",
                "Time-series database (InfluxDB, TimescaleDB)",
                "Real-time data streaming with Apache Kafka",
                "Data validation and quality control",
                "Missing data handling in sensor networks",
                "Network reliability and fault tolerance",
                "Practical project: 10-node farm sensor network"
            ]
        },
        {
            "filename": "yield-prediction.md",
            "title": "AI-Powered Yield Prediction",
            "sidebar_position": 3,
            "topic": "Build accurate yield prediction models combining satellite imagery, weather data, soil sensors, and historical records using ensemble ML and deep learning.",
            "key_points": [
                "Multi-source data integration (satellites, sensors, weather)",
                "Feature engineering from remote sensing (NDVI, EVI)",
                "Weather data APIs and forecasting",
                "Ensemble models combining multiple data sources",
                "Deep learning on multi-modal data",
                "Spatial and temporal modeling",
                "Pre-harvest yield estimation for logistics",
                "Practical project: County-level crop yield forecasting"
            ]
        },
        {
            "filename": "smart-irrigation.md",
            "title": "Automated Irrigation and Resource Management",
            "sidebar_position": 4,
            "topic": "Develop AI-driven irrigation control systems that optimize water usage using soil moisture, weather forecasts, and crop water requirements with reinforcement learning.",
            "key_points": [
                "Evapotranspiration and crop water needs",
                "Soil moisture sensor placement and interpretation",
                "Weather forecast integration",
                "Rule-based vs ML-based irrigation control",
                "Reinforcement learning for irrigation scheduling",
                "Fertigation: combined water and nutrient delivery",
                "Hardware: solenoid valves, pumps, controllers",
                "Practical project: Smart irrigation system with 30% water savings"
            ]
        },
        {
            "filename": "capstone-project.md",
            "title": "Capstone Project: Complete Smart Farm System",
            "sidebar_position": 5,
            "topic": "Integrate everything learned into a comprehensive smart farm platform with multi-sensor monitoring, disease detection, yield prediction, and automated control with web dashboard.",
            "key_points": [
                "System architecture: sensors, edge devices, cloud, web app",
                "Real-time monitoring dashboard (React + Chart.js)",
                "Disease detection from camera feeds",
                "Automated alerts and notifications",
                "Historical data analysis and trends",
                "Yield prediction and harvest planning",
                "Automated irrigation control integration",
                "Deployment: Docker, cloud hosting, mobile access",
                "Future enhancements: scaling to commercial farms"
            ]
        }
    ]
}

async def generate_lesson_content(client, module_name, lesson_info):
    """Generate comprehensive lesson content using Groq AI."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            prompt = f"""Create a comprehensive, educational markdown lesson for a university-level textbook on "AI Revolution in Plant Biotechnology".

Module: {module_name}
Title: {lesson_info['title']}
Topic: {lesson_info['topic']}

Key points to cover:
{chr(10).join(f"- {point}" for point in lesson_info['key_points'])}

Requirements:
1. Start with frontmatter:
---
sidebar_position: {lesson_info['sidebar_position']}
---

2. Include these sections:
   - Introduction with real-world motivation
   - Core concepts with clear explanations
   - Multiple code examples using Python (scikit-learn, TensorFlow/PyTorch, pandas, numpy)
   - Practical applications in agriculture/plant science
   - Best practices and common pitfalls
   - Hands-on example or mini-project
   - Summary table or checklist
   - Next steps and further reading

3. Writing style:
   - Clear, engaging, educational tone
   - Practical examples from agriculture/plant biotechnology
   - Code examples that actually work and are well-commented
   - Use tables, lists, and formatting for readability
   - Include specific plant examples (wheat, rice, tomato, etc.)
   - Add emojis sparingly (🌱 💡 ⚠️) only where appropriate

4. Code quality:
   - All code must be runnable and practical
   - Include imports and setup
   - Add comments explaining key steps
   - Show expected outputs

5. Length: Comprehensive (2000-3000 words minimum)

Generate the complete lesson content now:"""

            response = await client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{
                    "role": "system",
                    "content": "You are an expert educator in AI, plant biotechnology, and precision agriculture. "
                              "Create comprehensive, practical, code-rich lessons that teach both theory and implementation."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.4,
                max_tokens=8000
            )

            await asyncio.sleep(2)  # Rate limiting
            return response.choices[0].message.content

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"      Retry {attempt + 1}/{max_retries} after error: {str(e)[:100]}")
                await asyncio.sleep(5 * (attempt + 1))
            else:
                print(f"      ✗ Error after {max_retries} attempts: {e}")
                return None

async def generate_all_lessons():
    """Generate all missing lesson content."""
    client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
    docs_dir = Path("../website/docs")

    print("🌱 Starting AI-powered lesson generation for Plant Biotechnology...")
    print(f"📁 Target directory: {docs_dir.absolute()}\n")

    total_lessons = sum(len(lessons) for lessons in DOCS_STRUCTURE.values())
    current = 0

    for module_name, lessons in DOCS_STRUCTURE.items():
        module_dir = docs_dir / module_name
        module_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📚 Module: {module_name}")
        print(f"   Creating {len(lessons)} lessons...\n")

        for lesson_info in lessons:
            current += 1
            lesson_path = module_dir / lesson_info['filename']

            # Skip if already exists
            if lesson_path.exists():
                print(f"   [{current}/{total_lessons}] ⏭️  Skipping {lesson_info['filename']} (already exists)")
                continue

            print(f"   [{current}/{total_lessons}] 🤖 Generating {lesson_info['filename']}...")
            print(f"      Title: {lesson_info['title']}")

            content = await generate_lesson_content(client, module_name, lesson_info)

            if content:
                lesson_path.write_text(content, encoding='utf-8')
                print(f"      ✅ Created successfully!\n")
            else:
                print(f"      ⚠️  Failed to generate, skipping...\n")

    print("\n" + "="*60)
    print("✅ Base documentation generation complete!")
    print(f"📁 Location: {docs_dir.absolute()}")
    print(f"📊 Total lessons: {total_lessons}")
    print("\n🎯 Next steps:")
    print("   1. Review generated content")
    print("   2. Run generate_docs.py for personalization")
    print("   3. Run generate_urdu_docs.py for translation")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(generate_all_lessons())
