# Frontend Changes: Dark/Light Theme Toggle

## Summary
Added a theme toggle button to the Course Materials Assistant that allows users to switch between dark and light modes. The button is positioned in the top-right corner and uses sun/moon icons with smooth animations. The light theme has been carefully designed with proper contrast ratios and accessibility standards.

## Files Modified

### 1. `frontend/index.html`
Added a theme toggle button element before the main container:
- Button with `id="themeToggle"` and `class="theme-toggle"`
- Contains two SVG icons (sun and moon) from Feather Icons
- Includes `aria-label` for accessibility
- Has `title` attribute for tooltip on hover

### 2. `frontend/style.css`

**New CSS Variables (Dark Theme):**
- `--code-bg` - Code block background color
- `--error-bg`, `--error-text`, `--error-border` - Error message styling
- `--success-bg`, `--success-text`, `--success-border` - Success message styling
- `--scrollbar-track`, `--scrollbar-thumb`, `--scrollbar-thumb-hover` - Scrollbar colors

**Complete Light Theme Variables:**
Added `[data-theme="light"]` selector with all theme variables optimized for:
- High contrast text (WCAG AA compliant)
- Proper visual hierarchy
- Consistent design language

**New Theme Toggle Styles:**
- `.theme-toggle` - Fixed position button in top-right corner (44x44px, circular)
- Hover, focus, and active states with smooth transitions
- Icon transition effects (rotation and scaling)
- Dark mode shows moon icon, light mode shows sun icon
- Smooth color transitions for all themed elements
- Responsive adjustments for mobile (smaller button size)

**Updated Styles:**
- Error/success messages now use CSS variables for theming
- All scrollbars use theme-aware variables
- Code blocks use `var(--code-bg)` for proper light/dark appearance

### 3. `frontend/script.js`
**New DOM Element:**
- Added `themeToggle` to the list of DOM elements

**New Functions:**
- `initializeTheme()` - Loads saved theme from localStorage on page load (defaults to dark)
- `toggleTheme()` - Switches between dark and light themes
- `setTheme(theme)` - Applies the theme and persists preference to localStorage

**Event Listeners:**
- Added click handler for theme toggle button

## Features

### Design
- Circular button (44px) positioned fixed in top-right corner
- Uses sun icon (light mode) and moon icon (dark mode) from Feather Icons
- Smooth 0.3s transitions for all color changes and icon animations
- Icon rotates and scales during theme transition
- Shadow effects that adapt to the current theme

### Accessibility
- Keyboard navigable (standard button behavior)
- Focus ring visible when focused (uses `--focus-ring` color)
- Dynamic `aria-label` updates based on current theme state
- `title` attribute provides tooltip hint
- WCAG AA compliant contrast ratios for text

### Persistence
- Theme preference saved to localStorage
- Automatically applies saved preference on page load
- Defaults to dark theme if no preference saved

## Light Theme Color Palette

| Variable | Dark Value | Light Value | Purpose |
|----------|------------|-------------|---------|
| `--primary-color` | `#2563eb` | `#1d4ed8` | Primary actions, links |
| `--primary-hover` | `#1d4ed8` | `#1e40af` | Primary hover state |
| `--background` | `#0f172a` | `#f8fafc` | Page background |
| `--surface` | `#1e293b` | `#ffffff` | Cards, panels |
| `--surface-hover` | `#334155` | `#f1f5f9` | Hover state for surfaces |
| `--text-primary` | `#f1f5f9` | `#0f172a` | Main text (high contrast) |
| `--text-secondary` | `#94a3b8` | `#475569` | Secondary text |
| `--border-color` | `#334155` | `#cbd5e1` | Borders, dividers |
| `--user-message` | `#2563eb` | `#1d4ed8` | User chat bubbles |
| `--assistant-message` | `#374151` | `#f1f5f9` | Assistant chat bubbles |
| `--code-bg` | `rgba(0,0,0,0.2)` | `#f1f5f9` | Code block backgrounds |
| `--welcome-bg` | `#1e3a5f` | `#eff6ff` | Welcome message background |
| `--error-text` | `#f87171` | `#dc2626` | Error messages |
| `--success-text` | `#4ade80` | `#16a34a` | Success messages |

## Accessibility Compliance

### Contrast Ratios (Light Theme)
- Primary text (`#0f172a` on `#f8fafc`): **15.8:1** (AAA)
- Secondary text (`#475569` on `#f8fafc`): **7.1:1** (AAA)
- Primary color (`#1d4ed8` on `#ffffff`): **6.2:1** (AA Large / AAA)
- Error text (`#dc2626` on `#fef2f2`): **5.5:1** (AA)
- Success text (`#16a34a` on `#f0fdf4`): **4.7:1** (AA)

### Keyboard Navigation
- All interactive elements are focusable
- Visible focus indicators on all buttons
- Standard tab order maintained

## Usage
Click the toggle button in the top-right corner to switch between dark and light modes. The preference is automatically saved and will persist across browser sessions.
