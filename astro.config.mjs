// @ts-check
import { defineConfig } from 'astro/config';

import tailwindcss from '@tailwindcss/vite';

// https://astro.build/config
export default defineConfig({
  site: 'https://ldazrz.com',
  base: '',
  redirects: {
    '/projects': '/articles',
    '/projects/understanding-dwarf-star': '/articles/understanding-dwarf-star',
    '/projects/ski-telemetry': '/articles/ski-telemetry',
  },
  vite: {
    plugins: [tailwindcss()]
  }
});