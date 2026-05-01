# Web Demo for "Code Not Text" Research

A modern, interactive web demonstration of the cross-domain limits of hand-crafted CoT-surface features research.

## 🎯 Purpose

This demo is designed for showcasing the research findings on a personal website's tech section. It presents the key discoveries in an engaging, visually appealing format that's accessible to both technical and non-technical audiences.

## 📁 Structure

```
web_demo/
├── index.html          # Main demo page
├── README.md           # This file
├── DEPLOYMENT.md       # Deployment guide
├── css/
│   └── style.css       # Modern CSS with animations
├── js/
│   ├── charts.js       # Chart.js visualizations
│   └── animations.js   # Interactive effects
└── images/             # Placeholder for future images
```

## 🚀 Quick Start

### Local Development

1. **Simple HTTP Server (Python)**
```bash
cd web_demo
python -m http.server 8000
# Open http://localhost:8000
```

2. **Node.js HTTP Server**
```bash
npx http-server -p 8000
# Open http://localhost:8000
```

3. **VS Code Live Server**
- Install "Live Server" extension
- Right-click `index.html` → "Open with Live Server"

## 🎨 Features

### Interactive Elements
- **Animated Domain Cards**: Hover effects on math/science/coding results
- **Dynamic Charts**: Real-time Chart.js visualizations
- **Scroll Animations**: Smooth reveal effects as you scroll
- **Keyboard Navigation**: Use arrow keys to navigate sections
- **Responsive Design**: Works on desktop, tablet, and mobile

### Visual Highlights
- **Gradient Backgrounds**: Modern purple-to-blue theme
- **Performance Comparison Charts**: Clear AUROC comparisons
- **Method Breakdown**: Visual explanation of 5 convergent methods
- **Real-world Impact**: Best-of-N reranking results

### Accessibility
- Semantic HTML structure
- Keyboard navigation support
- Responsive text sizing
- High contrast colors

## 🌐 Deployment Options

### 1. GitHub Pages
Free hosting directly from your repository:

```bash
# Create gh-pages branch
git checkout --orphan gh-pages
git add web_demo/*
git commit -m "Deploy web demo"
git push origin gh-pages
```

### 2. Personal Website Integration

#### Option A: Iframe Embedding
```html
<iframe
    src="path/to/web_demo/index.html"
    width="100%"
    height="800px"
    frameborder="0">
</iframe>
```

#### Option B: Direct Integration
Copy the CSS and HTML sections into your existing tech page template.

#### Option C: Standalone Page
Deploy as a separate page: `yourwebsite.com/tech/code-not-text`

### 3. Netlify/Vercel
Drag and drop the `web_demo` folder to:
- [Netlify Drop](https://app.netlify.com/drop)
- [Vercel](https://vercel.com/new)

## 🎯 Key Sections

1. **Hero Section**: Eye-catching title and author info
2. **Key Finding**: Three domain comparison (Math: 0.958, Science: 0.799, Coding: 0.434)
3. **Performance Chart**: Visual AUROC comparison
4. **Five Methods**: Interactive method cards
5. **Implications**: Why this matters for verification
6. **Reranking Impact**: Real-world Best-of-N results
7. **Clarifications**: What we do/don't claim
8. **Technical Details**: Model and data specifications

## 🛠️ Customization

### Colors
Edit `css/style.css` variables:
```css
:root {
    --math-color: #1f77b4;
    --science-color: #2ca02c;
    --coding-color: #d62728;
}
```

### Charts
Modify data in `js/charts.js`:
```javascript
data: [0.982, 0.841, 0.407]  // AUROC values
```

### Content
Edit `index.html` to update:
- Author information
- Links to GitHub/PDF
- Contact details

## 📱 Browser Support

- ✅ Chrome/Edge (latest)
- ✅ Firefox (latest)
- ✅ Safari (latest)
- ✅ Mobile browsers

## 🎨 Design Philosophy

1. **Scannability**: Key metrics visible immediately
2. **Progressive Disclosure**: Details on hover/scroll
3. **Visual Hierarchy**: Size and color guide attention
4. **Modern Aesthetics**: Gradients, shadows, smooth animations
5. **Performance**: Minimal external dependencies

## 📊 Data Sources

All data comes from the research paper:
- **Model**: DeepSeek-R1-0528-Qwen3-8B
- **Domains**: Math (AIME/BRUMO/HMMT), Science (GPQA), Coding (LiveCodeBench-v5)
- **Metrics**: AUC-of-AUROC, AUROC@100%, Best-of-N pass@1

## 🔧 Dependencies

- **Chart.js 4.4.0** (via CDN for charts)
- **Modern CSS** (Grid, Flexbox, Custom Properties)
- **Vanilla JavaScript** (No framework required)

## 📝 License

Same as parent project: Apache 2.0

## 🤝 Contributing

To update the demo:
1. Modify content in `index.html`
2. Update styles in `css/style.css`
3. Adjust charts in `js/charts.js`
4. Test locally before deploying

## 📧 Contact

For questions about the demo or research:
Yuhan Chi · Fudan University · yhchi25@m.fudan.edu.cn

## 🌟 Acknowledgments

- Chart.js for beautiful visualizations
- Modern CSS features for smooth animations
- Research community for feedback and insights
