import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import RAGChatbot from '../components/RAGChatbot';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <h1 className={styles.heroTitle}>{siteConfig.title}</h1>
        <p className={styles.heroSubtitle}>{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Start Learning 🚀
          </Link>
          <Link
            className="button button--outline button--secondary button--lg"
            to="https://github.com/biologist01/-ai-in-plantbiotics-book">
            View on GitHub ⭐
          </Link>
        </div>
      </div>
    </header>
  );
}

function Feature({title, description, icon}) {
  return (
    <div className={clsx('col col--4')}>
      <div className={styles.featureCard}>
        <div className={styles.featureIcon}>
          {icon}
        </div>
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function Home(): React.ReactElement {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="Master AI in Plant Biotechnology">
      <HomepageHeader />
      <main>
        <section className={styles.features}>
          <div className="container">
            <h2 className={styles.sectionTitle}>Course Modules</h2>
            <div className="row">
              <Feature 
                title="Module 1: ML in Agriculture" 
                description="Master the basics of Machine Learning applied to agricultural datasets. Learn classification, regression, and time-series analysis for crop yield prediction."
                icon="🤖"
              />
              <Feature 
                title="Module 2: Computer Vision" 
                description="Build systems that can 'see' plants. Implement CNNs for disease detection, leaf counting, and phenotyping."
                icon="👁️"
              />
              <Feature 
                title="Module 3: Genomics & CRISPR" 
                description="Dive into the code of life. Use AI to analyze genomic sequences and design CRISPR edits for crop improvement."
                icon="🧬"
              />
            </div>
            <div className="row" style={{marginTop: '2rem'}}>
              <Feature 
                title="Module 4: IoT & Smart Farming" 
                description="Connect the physical world. Deploy sensor networks and build smart irrigation systems powered by AI."
                icon="📡"
              />
              <Feature 
                title="AI Chatbot Assistant" 
                description="Stuck on a concept? Our RAG-powered AI assistant is trained on the entire textbook to help you learn faster."
                icon="💬"
              />
              <Feature 
                title="Capstone Project" 
                description="Apply everything you've learned to build a comprehensive AI solution for a real-world agricultural problem."
                icon="🏆"
              />
            </div>
          </div>
        </section>
        
        <section className={styles.ctaSection}>
          <div className="container">
            <h2>Ready to Transform Agriculture?</h2>
            <p>Join the revolution in Plant Biotechnology with Artificial Intelligence.</p>
            <Link
              className="button button--secondary button--lg"
              to="/docs/intro">
              Begin Your Journey →
            </Link>
          </div>
        </section>
      </main>
      <RAGChatbot />
    </Layout>
  );
}
