# AI Revolution in Plant Biotechnology - Documentation

This documentation site contains the complete book "AI Revolution in Plant Biotechnology" built with [Docusaurus](https://docusaurus.io/), a modern static website generator.

## Installation

```bash
npm install
```

## Local Development

```bash
npm start
```

This command starts a local development server and opens a browser window. Most changes are reflected live without having to restart the server.

## Build

```bash
npm run build
```

This command generates static content into the `build` directory and can be served using any static content hosting service.

## Deployment

### GitHub Pages

The site is configured for deployment to GitHub Pages. To deploy:

Using SSH:
```bash
USE_SSH=true npm run deploy
```

Not using SSH:
```bash
GIT_USER=biologist01 npm run deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.

### Vercel

The site can be deployed to Vercel by connecting your GitHub repository. Make sure to set the build command to `npm run build` and the output directory to `build`.

### Other Platforms

The site can be deployed to any static hosting service by uploading the contents of the `build` directory.

## Configuration Notes

The site is pre-configured for deployment with:
- `url`: https://biologist01.github.io
- `baseUrl`: /
- `organizationName`: biologist01
- `projectName`: ai-in-plantbiotics-book
