# Deployment Guide for Code Not Text Web Demo

## 🚀 Quick Deployment Options

### Option 1: GitHub Pages (Recommended for Free Hosting)

#### Step 1: Prepare Repository
```bash
# Navigate to your repository
cd /path/to/code-not-text

# Create gh-pages branch if it doesn't exist
git checkout --orphan gh-pages
git rm -rf .
```

#### Step 2: Copy Demo Files
```bash
# Copy only the web_demo folder content
cp -r web_demo/* .
cp web_demo/.gitignore . 2>/dev/null || true

# Commit and push
git add .
git commit -m "Deploy web demo to GitHub Pages"
git push origin gh-pages
```

#### Step 3: Enable GitHub Pages
1. Go to repository Settings → Pages
2. Source: Deploy from a branch
3. Branch: `gh-pages` / `root`
4. Save

Your demo will be live at: `https://your-username.github.io/code-not-text/`

### Option 2: Personal Website Integration

#### A. Subdirectory Deployment
Deploy to existing website: `yourwebsite.com/tech/code-not-text/`

```bash
# Copy web_demo to your website folder
cp -r web_demo /path/to/your-website/tech/code-not-text

# Update paths in index.html if needed
# (Most paths are relative, so should work as-is)
```

#### B. Iframe Embedding
Embed in existing tech page:

```html
<!-- In your tech page -->
<div class="research-demo">
    <h2>Code Not Text: Cross-Domain Limits of CoT Features</h2>
    <iframe
        src="/demos/code-not-text/index.html"
        width="100%"
        height="1200px"
        frameborder="0"
        scrolling="yes">
    </iframe>
</div>

<style>
.research-demo {
    margin: 2rem 0;
}
.research-demo iframe {
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
</style>
```

#### C. WordPress Integration
Use Custom HTML Block:

```html
<iframe
    src="https://your-username.github.io/code-not-text/"
    width="100%"
    height="1000px"
    frameborder="0">
</iframe>
```

### Option 3: Netlify Drop (Drag & Drop)

1. Go to [Netlify Drop](https://app.netlify.com/drop)
2. Drag the entire `web_demo` folder
3. Get instant URL: `https://random-name.netlify.app`
4. Custom domain: `codenotext.yourdomain.com`

### Option 4: Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy from web_demo folder
cd web_demo
vercel --prod
```

## 🛠️ Configuration Updates

### Update Contact Information

Edit `index.html` footer section:

```html
<footer class="footer">
    <div class="container">
        <p>© 2026 Your Name · Your Institution ·
        <a href="https://github.com/your-username/code-not-text">code-not-text</a>
        </p>
        <p class="license">Apache 2.0 License</p>
    </div>
</footer>
```

### Update Links in CTA Section

```html
<div class="cta-links">
    <a href="https://github.com/your-username/code-not-text" class="cta-button primary">
        📦 GitHub Repository
    </a>
    <a href="path/to/paper.pdf" class="cta-button secondary">
        📄 Full Paper (PDF)
    </a>
    <a href="mailto:your-email@domain.com" class="cta-button secondary">
        ✉️ Contact Author
    </a>
</div>
```

## 📱 Mobile Optimization

The demo is responsive by default, but test on mobile:

```bash
# Chrome DevTools mobile testing
# F12 → Toggle device toolbar → Test different viewports
```

## 🔍 SEO Optimization

Add meta tags to `index.html`:

```html
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Code Not Text - Cross-Domain Limits of CoT Quality Measurement</title>

    <!-- SEO Meta Tags -->
    <meta name="description" content="Research showing hand-crafted CoT features work in math, partly in science, and fail in coding correctness prediction.">
    <meta name="keywords" content="chain-of-thought, code verification, machine learning, NLP, cross-domain">
    <meta name="author" content="Yuhan Chi">

    <!-- Open Graph / Social Media -->
    <meta property="og:title" content="Code Not Text - Cross-Domain Limits of CoT Features">
    <meta property="og:description" content="Five methods converge: CoT quality measurement breaks down in coding">
    <meta property="og:image" content="https://your-domain.com/preview-image.png">
    <meta property="og:url" content="https://your-domain.com/tech/code-not-text">
    <meta property="og:type" content="website">

    <!-- Twitter Card -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="Code Not Text">
    <meta name="twitter:description" content="Cross-domain limits of hand-crafted CoT features">
    <meta name="twitter:image" content="https://your-domain.com/preview-image.png">

    <link rel="stylesheet" href="css/style.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
```

## 🎨 Customization Examples

### Change Color Scheme

Edit `css/style.css`:

```css
:root {
    /* Purple to Blue gradient (current) */
    --gradient-start: #667eea;
    --gradient-end: #764ba2;

    /* Alternative: Green to Teal */
    --gradient-start: #11998e;
    --gradient-end: #38ef7d;

    /* Alternative: Orange to Red */
    --gradient-start: #f12711;
    --gradient-end: #f5af19;
}
```

### Modify Chart Data

Edit `js/charts.js`:

```javascript
// Performance Comparison Chart
data: [0.982, 0.841, 0.407],  // Update these values
labels: ['Math', 'Science', 'Coding'],  // Update labels
```

### Add Custom Analytics

```html
<!-- Add to index.html before </head> -->
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

## 🚦 Pre-Deployment Checklist

- [ ] Contact information updated
- [ ] GitHub links corrected
- [ ] Paper PDF link working
- [ ] Test on mobile devices
- [ ] Test in different browsers
- [ ] Check load times (optimize images if needed)
- [ ] Verify all external links
- [ ] Add analytics if desired
- [ ] Test keyboard navigation
- [ ] Verify accessibility

## 📊 Performance Optimization

### Lazy Load Charts
```javascript
// In js/charts.js, replace automatic init with:
document.addEventListener('DOMContentLoaded', function() {
    // Only initialize charts when section is visible
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                initCharts();
                observer.disconnect();
            }
        });
    });

    observer.observe(document.querySelector('.comparison'));
});
```

### Optimize Images
```bash
# If you add images, optimize them
# Install: npm install -g imagemin-cli
imagemin images/* --out-dir=images/optimized
```

## 🐛 Troubleshooting

### Charts Not Displaying
- Check browser console for errors
- Verify Chart.js CDN is accessible
- Ensure canvas elements exist in DOM

### Styles Not Loading
- Verify CSS file path is correct
- Check browser developer tools network tab
- Clear browser cache

### Mobile Layout Issues
- Test in responsive design mode
- Check viewport meta tag
- Verify CSS media queries

## 🔗 URL Structure Suggestions

Choose a clean, memorable URL:

```
yourwebsite.com/tech/code-not-text
yourwebsite.com/research/coy-quality-limits
yourwebsite.com/projects/cross-domain-cot
```

## 📈 Analytics Integration

### Simple Page View Tracking
```javascript
// Add to js/animations.js
document.addEventListener('DOMContentLoaded', function() {
    // Track page view
    if (window.gtag) {
        gtag('event', 'page_view', {
            'page_title': 'Code Not Text Demo',
            'page_path': '/tech/code-not-text'
        });
    }

    // Track chart interactions
    document.querySelectorAll('.domain-card').forEach(card => {
        card.addEventListener('click', function() {
            if (window.gtag) {
                gtag('event', 'click', {
                    'event_category': 'engagement',
                    'event_label': 'domain_card_' + this.querySelector('h3').textContent.toLowerCase()
                });
            }
        });
    });
});
```

## 🎯 Post-Deployment

1. **Test Live URL**: Check all functionality works
2. **Monitor Performance**: Use tools like PageSpeed Insights
3. **Gather Feedback**: Share with colleagues for testing
4. **Update Links**: Add to your CV, research profiles, etc.
5. **Document**: Keep this file updated with any changes

## 📞 Support

For deployment issues:
- Check browser console for errors
- Verify file paths and permissions
- Test in different browsers
- Review deployment platform documentation

Enjoy showcasing your research! 🚀
