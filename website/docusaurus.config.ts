import { themes as prismThemes } from 'prism-react-renderer';
import type { Config } from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// Plant Biotechnology AI Textbook configuration
const config: Config = {
  title: 'AI Revolution in Plant Biotechnology',
  tagline: 'Transforming Agriculture with Artificial Intelligence',
  favicon: 'img/favicon.ico',

  // For GitHub Pages, ensure `baseUrl` includes the repo name
  url: 'https://biologist01.github.io', // for GitHub Pages root
  baseUrl: '/-ai-in-plantbiotics-book/', // Add repo name here

  organizationName: 'biologist01', // Your GitHub username
  projectName: '-ai-in-plantbiotics-book', // Your repository name

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'], // You can add 'ur' here if you plan to support Urdu translation
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts', // Path to your sidebar file
          editUrl: 'https://github.com/biologist01/-ai-in-plantbiotics-book/tree/main/website/',
        },
        blog: false, // Disable blog (if you don't need it)
        theme: {
          customCss: './src/css/custom.css', // Path to custom styles (if any)
        },
      } satisfies Preset.Options,
    ],
  ],

  plugins: [
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'docs-software',
        path: 'docs-software',
        routeBasePath: 'docs-software', // Custom route
        sidebarPath: './sidebars.ts',
        editUrl: 'https://github.com/biologist01/-ai-in-plantbiotics-book/tree/main/website/',
      },
    ],
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'docs-hardware',
        path: 'docs-hardware',
        routeBasePath: 'docs-hardware', // Custom route
        sidebarPath: './sidebars.ts',
        editUrl: 'https://github.com/biologist01/-ai-in-plantbiotics-book/tree/main/website/',
      },
    ],
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'docs-urdu',
        path: 'docs-urdu',
        routeBasePath: 'docs-urdu', // Custom route
        sidebarPath: './sidebars-urdu.ts', // Ensure this file exists
        editUrl: 'https://github.com/biologist01/-ai-in-plantbiotics-book/tree/main/website/',
      },
    ],
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'docs-urdu-software',
        path: 'docs-urdu-software',
        routeBasePath: 'docs-urdu-software', // Custom route
        sidebarPath: './sidebars-urdu.ts', // Ensure this file exists
        editUrl: 'https://github.com/biologist01/-ai-in-plantbiotics-book/tree/main/website/',
      },
    ],
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'docs-urdu-hardware',
        path: 'docs-urdu-hardware',
        routeBasePath: 'docs-urdu-hardware', // Custom route
        sidebarPath: './sidebars-urdu.ts', // Ensure this file exists
        editUrl: 'https://github.com/biologist01/-ai-in-plantbiotics-book/tree/main/website/',
      },
    ],
  ],

  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',
    navbar: {
      title: 'Plant AI',
      logo: {
        alt: 'Plant AI Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar', // Ensure this sidebar ID exists
          position: 'left',
          label: 'Textbook',
        },
        {
          href: 'https://github.com/biologist01/-ai-in-plantbiotics-book',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Textbook',
          items: [
            {
              label: 'Introduction',
              to: '/docs/intro',
            },
            {
              label: 'Get Started',
              to: '/docs/module-1/ml-intro',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/biologist01/-ai-in-plantbiotics-book',
            },
            {
              label: 'Panaversity',
              href: 'https://panaversity.org',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Plant AI Textbook. Created by <strong>Fatima Amir</strong> for Panaversity Hackathon.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'json', 'yaml', 'cpp'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
