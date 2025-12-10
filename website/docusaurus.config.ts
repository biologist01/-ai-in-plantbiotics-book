import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// Plant Biotechnology AI Textbook configuration
const config: Config = {
  title: 'AI Revolution in Plant Biotechnology',
  tagline: 'Transforming Agriculture with Artificial Intelligence',
  favicon: 'img/favicon.ico',

  url: 'https://biologist01.github.io',
  baseUrl: '/-ai-in-plantbiotics-book/',

  organizationName: 'biologist01',
  projectName: '-ai-in-plantbiotics-book',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl:
            'https://github.com/biologist01/-ai-in-plantbiotics-book/tree/main/website/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
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
        routeBasePath: 'docs-software',
        sidebarPath: './sidebars.ts',
        editUrl: 'https://github.com/biologist01/-ai-in-plantbiotics-book/tree/main/website/',
      },
    ],
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'docs-hardware',
        path: 'docs-hardware',
        routeBasePath: 'docs-hardware',
        sidebarPath: './sidebars.ts',
        editUrl: 'https://github.com/biologist01/-ai-in-plantbiotics-book/tree/main/website/',
      },
    ],
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'docs-urdu',
        path: 'docs-urdu',
        routeBasePath: 'docs-urdu',
        sidebarPath: './sidebars-urdu.ts',
        editUrl: 'https://github.com/biologist01/-ai-in-plantbiotics-book/tree/main/website/',
      },
    ],
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'docs-urdu-software',
        path: 'docs-urdu-software',
        routeBasePath: 'docs-urdu-software',
        sidebarPath: './sidebars-urdu.ts',
        editUrl: 'https://github.com/biologist01/-ai-in-plantbiotics-book/tree/main/website/',
      },
    ],
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'docs-urdu-hardware',
        path: 'docs-urdu-hardware',
        routeBasePath: 'docs-urdu-hardware',
        sidebarPath: './sidebars-urdu.ts',
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
          sidebarId: 'tutorialSidebar',
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
