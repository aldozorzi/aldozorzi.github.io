// Theme definitions — only colors actually used in the site
export const themes = {
  catppuccin: {
    mode: "dark",
    background: "#1e1e2e",
    muted: "#585b70",
    foreground: "#cdd6f4",
    brightForeground: "#cdd6f4",
    accent: "#87a7da",
  },
  "catppuccin-latte": {
    mode: "light",
    background: "#f3f5ef",
    muted: "#acb0be",
    foreground: "#4c4f69",
    brightForeground: "#4c4f69",
    accent: "#305399",
  },
  "coffee": {
    mode: "dark",
    background: "#201200",
    muted: "#af9a47",
    foreground: "#ece0b2",
    brightForeground: "#f5efda",
    accent: "#ee9b00",
  },
  ethereal: {
    mode: "dark",
    background: "#060B1E",
    muted: "#5f4a80",
    foreground: "#ffcead",
    brightForeground: "#ffcead",
    accent: "#a07dd9",
  },
  everforest: {
    mode: "dark",
    background: "#2d353b",
    muted: "#475258",
    foreground: "#d3c6aa",
    brightForeground: "#d3c6aa",
    accent: "#7fbbb3",
  },
  "flexoki-light": {
    mode: "light",
    background: "#FFFCF0",
    muted: "#B7B5AC",
    foreground: "#100F0F",
    brightForeground: "#100F0F",
    accent: "#205EA6",
  },
  gruvbox: {
    mode: "dark",
    background: "#282828",
    muted: "#665c54",
    foreground: "#d4be98",
    brightForeground: "#d4be98",
    accent: "#7daea3",
  },
  hackerman: {
    mode: "dark",
    background: "#0B0C16",
    muted: "#4c5888",
    foreground: "#ddf7ff",
    brightForeground: "#ddf7ff",
    accent: "#82FB9C",
  },
  kanagawa: {
    mode: "dark",
    background: "#1f1f28",
    muted: "#54546D",
    foreground: "#dcd7ba",
    brightForeground: "#dcd7ba",
    accent: "#dcd7ba",
  },
  "last-horizon": {
    mode: "dark",
    background: "#0c0b0c",
    muted: "#584e51",
    foreground: "#FAFCFB",
    brightForeground: "#e2dddc",
    accent: "#b59790",
  },
  lumon: {
    mode: "dark",
    background: "#16242d",
    muted: "#304860",
    foreground: "#d6e2ee",
    brightForeground: "#f2fcff",
    accent: "#8bc9eb",
  },
  lupine: {
    mode: "light",
    background: "#fafafa",
    muted: "#9e9e9e",
    foreground: "#212121",
    brightForeground: "#000000",
    accent: "#607ed3",
  },
  "matte-black": {
    mode: "dark",
    background: "#121212",
    muted: "#333333",
    foreground: "#bebebe",
    brightForeground: "#bebebe",
    accent: "#e68e0d",
  },
  miasma: {
    mode: "dark",
    background: "#222222",
    muted: "#666666",
    foreground: "#c2c2b0",
    brightForeground: "#c2c2b0",
    accent: "#78824b",
  },
  nord: {
    mode: "dark",
    background: "#2e3440",
    muted: "#4c566a",
    foreground: "#d8dee9",
    brightForeground: "#d8dee9",
    accent: "#81a1c1",
  },
  "osaka-jade": {
    mode: "dark",
    background: "#111c18",
    muted: "#53685B",
    foreground: "#C1C497",
    brightForeground: "#F7E8B2",
    accent: "#509475",
  },
  "retro-82": {
    mode: "dark",
    background: "#05182e",
    muted: "#2a6b78",
    foreground: "#f6dcac",
    brightForeground: "#f6dcac",
    accent: "#f77e1b",
  },
  ristretto: {
    mode: "dark",
    background: "#2c2525",
    muted: "#72696a",
    foreground: "#e6d9db",
    brightForeground: "#e6d9db",
    accent: "#f38d70",
  },
  "rose-pine": {
    mode: "light",
    background: "#faf4ed",
    muted: "#cecacd",
    foreground: "#575279",
    brightForeground: "#575279",
    accent: "#56949f",
  },
  solitude: {
    mode: "dark",
    background: "#101315",
    muted: "#4b4e55",
    foreground: "#cacccc",
    brightForeground: "#a5aeb4",
    accent: "#798186",
  },
  "tokyo-night": {
    mode: "dark",
    background: "#1a1b26",
    muted: "#414868",
    foreground: "#a9b1d6",
    brightForeground: "#c0caf5",
    accent: "#ea00ff",
  },
  vantablack: {
    mode: "dark",
    background: "#000000",
    muted: "#7a7a7a",
    foreground: "#ffffff",
    brightForeground: "#ffffff",
    accent: "#8d8d8d",
  },
  white: {
    mode: "light",
    background: "#ffffff",
    muted: "#808080",
    foreground: "#000000",
    brightForeground: "#000000",
    accent: "#6e6e6e",
  },
};

type ThemeName = keyof typeof themes;
type Theme = (typeof themes)[ThemeName];

const STORAGE_KEY = "ldazrz-theme";
const EXPIRY_HOURS = 12;

interface StoredTheme {
  name: ThemeName;
  timestamp: number;
}

function getStoredTheme(): ThemeName | null {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return null;

    const parsed: StoredTheme = JSON.parse(stored);
    const elapsed = Date.now() - parsed.timestamp;
    const maxAge = EXPIRY_HOURS * 60 * 60 * 1000;

    if (elapsed > maxAge) {
      localStorage.removeItem(STORAGE_KEY);
      return null;
    }

    return parsed.name;
  } catch {
    return null;
  }
}

function storeTheme(name: ThemeName): void {
  const data: StoredTheme = { name, timestamp: Date.now() };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
}

function getRandomTheme(): ThemeName {
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const filtered = (Object.keys(themes) as ThemeName[]).filter(
    (k) => themes[k].mode === (prefersDark ? "dark" : "light")
  );
  return filtered[Math.floor(Math.random() * filtered.length)];
}

function applyTheme(theme: Theme): void {
  const root = document.documentElement;
  root.style.setProperty("--theme-background", theme.background);
  root.style.setProperty("--theme-muted", theme.muted);
  root.style.setProperty("--theme-foreground", theme.foreground);
  root.style.setProperty("--theme-bright-foreground", theme.brightForeground);
  root.style.setProperty("--theme-accent", theme.accent);
}

export function initTheme(): void {
  const stored = getStoredTheme();
  const name = stored || getRandomTheme();

  if (!stored) {
    storeTheme(name);
  }

  applyTheme(themes[name]);
}
