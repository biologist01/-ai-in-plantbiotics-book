import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: Machine Learning Foundations',
      items: [
        'module-1/ml-intro',
        'module-1/data-preprocessing',
        'module-1/classification-models',
        'module-1/regression-models',
        'module-1/time-series',
        'module-1/ml-project',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Computer Vision in Agriculture',
      items: [
        'module-2/cv-intro',
        'module-2/image-processing',
        'module-2/deep-learning-cnn',
        'module-2/object-detection',
        'module-2/cv-project',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: Genomics & CRISPR AI',
      items: [
        'module-3/genomics-intro',
        'module-3/sequence-analysis',
        'module-3/genomic-selection',
        'module-3/crispr-ai',
        'module-3/genomics-project',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: IoT & Smart Farming',
      items: [
        'module-4/iot-intro',
        'module-4/sensor-networks',
        'module-4/smart-irrigation',
        'module-4/yield-prediction',
        'module-4/capstone-project',
      ],
    },
  ],
};

export default sidebars;
