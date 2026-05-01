// Interactive Animations and Effects for Code Not Text Demo

document.addEventListener('DOMContentLoaded', function() {

    // Smooth scroll for navigation
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href !== '#' && document.querySelector(href)) {
                e.preventDefault();
                const target = document.querySelector(href);
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Intersection Observer for scroll animations
    const scrollRevealOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const scrollRevealObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
                scrollRevealObserver.unobserve(entry.target);
            }
        });
    }, scrollRevealOptions);

    // Apply scroll animations to sections
    document.querySelectorAll('section').forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(30px)';
        section.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
        scrollRevealObserver.observe(section);
    });

    // Domain card hover effects with enhanced interactivity
    document.querySelectorAll('.domain-card').forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-10px) scale(1.02)';
            this.style.boxShadow = '0 20px 40px rgba(0,0,0,0.2)';

            // Pulse effect on the metric
            const metric = this.querySelector('.metric .value');
            if (metric) {
                metric.style.transform = 'scale(1.1)';
                metric.style.transition = 'transform 0.3s ease';
            }
        });

        card.addEventListener('mouseleave', function() {
            this.style.transform = '';
            this.style.boxShadow = '';

            const metric = this.querySelector('.metric .value');
            if (metric) {
                metric.style.transform = '';
            }
        });

        // Click to show more details
        card.addEventListener('click', function() {
            const detail = this.querySelector('.detail');
            if (detail) {
                detail.style.transition = 'all 0.3s ease';
                detail.style.transform = 'scale(1.05)';
                setTimeout(() => {
                    detail.style.transform = '';
                }, 300);
            }
        });
    });

    // Method cards stagger animation
    const methodCards = document.querySelectorAll('.method-card');
    methodCards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
        card.style.animation = 'fadeInUp 0.6s ease-out forwards';
        card.style.opacity = '0';
    });

    // Implication cards hover effects
    document.querySelectorAll('.implication-card').forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px) rotateX(5deg)';
            this.style.boxShadow = '0 15px 30px rgba(0,0,0,0.2)';
        });

        card.addEventListener('mouseleave', function() {
            this.style.transform = '';
            this.style.boxShadow = '';
        });
    });

    // CTA button ripple effect
    document.querySelectorAll('.cta-button').forEach(button => {
        button.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;

            ripple.style.cssText = `
                position: absolute;
                width: ${size}px;
                height: ${size}px;
                left: ${x}px;
                top: ${y}px;
                background: rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                transform: scale(0);
                animation: ripple 0.6s ease-out;
                pointer-events: none;
            `;

            this.style.position = 'relative';
            this.style.overflow = 'hidden';
            this.appendChild(ripple);

            setTimeout(() => ripple.remove(), 600);
        });
    });

    // Add ripple animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes ripple {
            to {
                transform: scale(4);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);

    // Parallax effect for hero section
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        const hero = document.querySelector('.hero');
        if (hero) {
            const parallaxSpeed = 0.5;
            hero.style.transform = `translateY(${scrolled * parallaxSpeed}px)`;
        }
    });

    // Progress indicator for page reading
    function createProgressBar() {
        const progressBar = document.createElement('div');
        progressBar.id = 'reading-progress';
        progressBar.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            height: 3px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            width: 0%;
            z-index: 1000;
            transition: width 0.1s ease;
        `;
        document.body.appendChild(progressBar);

        window.addEventListener('scroll', () => {
            const windowHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
            const scrolled = (window.pageYOffset / windowHeight) * 100;
            progressBar.style.width = scrolled + '%';
        });
    }

    createProgressBar();

    // Add keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
            const currentSection = document.querySelector('section:hover') ||
                                  document.elementFromPoint(window.innerWidth / 2, window.innerHeight / 2)?.closest('section');

            if (currentSection) {
                const nextSection = currentSection.nextElementSibling;
                if (nextSection && nextSection.tagName === 'SECTION') {
                    nextSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }
        } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
            const currentSection = document.querySelector('section:hover') ||
                                  document.elementFromPoint(window.innerWidth / 2, window.innerHeight / 2)?.closest('section');

            if (currentSection) {
                const prevSection = currentSection.previousElementSibling;
                if (prevSection && prevSection.tagName === 'SECTION') {
                    prevSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }
        }
    });

    // Add subtle particle effect to hero section
    function createParticles() {
        const hero = document.querySelector('.hero');
        if (!hero) return;

        const particleCount = 20;

        for (let i = 0; i < particleCount; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';

            const size = Math.random() * 5 + 2;
            const posX = Math.random() * 100;
            const posY = Math.random() * 100;
            const delay = Math.random() * 5;
            const duration = Math.random() * 10 + 10;

            particle.style.cssText = `
                position: absolute;
                width: ${size}px;
                height: ${size}px;
                background: rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                left: ${posX}%;
                top: ${posY}%;
                animation: float ${duration}s ease-in-out ${delay}s infinite;
                pointer-events: none;
            `;

            hero.style.position = 'relative';
            hero.style.overflow = 'hidden';
            hero.appendChild(particle);
        }

        // Add float animation
        const floatStyle = document.createElement('style');
        floatStyle.textContent = `
            @keyframes float {
                0%, 100% { transform: translateY(0) rotate(0deg); opacity: 0.3; }
                50% { transform: translateY(-20px) rotate(180deg); opacity: 0.8; }
            }
        `;
        document.head.appendChild(floatStyle);
    }

    createParticles();

    // Add tooltip enhancement
    function enhanceTooltips() {
        const tooltipElements = document.querySelectorAll('[title], [data-tooltip]');

        tooltipElements.forEach(element => {
            element.addEventListener('mouseenter', function(e) {
                const text = this.getAttribute('title') || this.getAttribute('data-tooltip');

                if (!text) return;

                const tooltip = document.createElement('div');
                tooltip.className = 'custom-tooltip';
                tooltip.textContent = text;
                tooltip.style.cssText = `
                    position: absolute;
                    background: rgba(0, 0, 0, 0.8);
                    color: white;
                    padding: 8px 12px;
                    border-radius: 6px;
                    font-size: 14px;
                    pointer-events: none;
                    z-index: 1000;
                    white-space: nowrap;
                    opacity: 0;
                    transition: opacity 0.3s ease;
                `;

                document.body.appendChild(tooltip);

                const rect = this.getBoundingClientRect();
                tooltip.style.left = rect.left + rect.width / 2 - tooltip.offsetWidth / 2 + 'px';
                tooltip.style.top = rect.top - tooltip.offsetHeight - 10 + 'px';

                setTimeout(() => tooltip.style.opacity = '1', 10);

                this.addEventListener('mouseleave', function() {
                    tooltip.style.opacity = '0';
                    setTimeout(() => tooltip.remove(), 300);
                }, { once: true });
            });
        });
    }

    enhanceTooltips();

    // Performance monitoring (optional, for development)
    if (window.performance) {
        window.addEventListener('load', () => {
            const perfData = performance.timing;
            const pageLoadTime = perfData.loadEventEnd - perfData.navigationStart;
            console.log(`Page load time: ${pageLoadTime}ms`);
        });
    }

    // Add smooth reveal for claim boxes
    const claimBoxes = document.querySelectorAll('.claim-box');
    claimBoxes.forEach((box, index) => {
        box.style.animationDelay = `${index * 0.2}s`;
        box.style.animation = 'fadeInUp 0.8s ease-out forwards';
        box.style.opacity = '0';
    });

    // Highlight effect for technical specs
    document.querySelectorAll('.spec-item').forEach(item => {
        item.addEventListener('mouseenter', function() {
            this.style.background = 'rgba(102, 126, 234, 0.1)';
            this.style.transform = 'scale(1.02)';
            this.style.transition = 'all 0.3s ease';
        });

        item.addEventListener('mouseleave', function() {
            this.style.background = '';
            this.style.transform = '';
        });
    });

    // Add mobile touch support
    if ('ontouchstart' in window) {
        document.querySelectorAll('.domain-card, .method-card, .implication-card').forEach(card => {
            card.addEventListener('touchstart', function() {
                this.style.transform = 'scale(0.98)';
            });

            card.addEventListener('touchend', function() {
                this.style.transform = '';
            });
        });
    }

    console.log('🚀 Code Not Text Demo Initialized');
    console.log('💡 Tip: Use arrow keys to navigate between sections');
});
