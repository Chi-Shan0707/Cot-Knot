// Chart.js Configuration and Data
document.addEventListener('DOMContentLoaded', function() {

    // Performance Comparison Chart
    const performanceCtx = document.getElementById('performanceChart').getContext('2d');

    new Chart(performanceCtx, {
        type: 'bar',
        data: {
            labels: ['Math', 'Science', 'Coding'],
            datasets: [{
                label: 'AUROC @ 100%',
                data: [0.982, 0.841, 0.407],
                backgroundColor: [
                    'rgba(31, 119, 180, 0.8)',
                    'rgba(44, 160, 44, 0.8)',
                    'rgba(214, 39, 40, 0.8)'
                ],
                borderColor: [
                    'rgba(31, 119, 180, 1)',
                    'rgba(44, 160, 44, 1)',
                    'rgba(214, 39, 40, 1)'
                ],
                borderWidth: 2,
                borderRadius: 8,
                borderSkipped: false,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return 'AUROC: ' + context.parsed.y.toFixed(3);
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            },
            animation: {
                duration: 1500,
                easing: 'easeInOutQuart'
            }
        }
    });

    // Reranking Impact Chart
    const rerankingCtx = document.getElementById('rerankingChart').getContext('2d');

    new Chart(rerankingCtx, {
        type: 'bar',
        data: {
            labels: ['Math', 'Science', 'Coding'],
            datasets: [
                {
                    label: 'Random Selection',
                    data: [64.2, 60.2, 61.7],
                    backgroundColor: 'rgba(149, 165, 166, 0.6)',
                    borderColor: 'rgba(149, 165, 166, 1)',
                    borderWidth: 2,
                    borderRadius: 8,
                },
                {
                    label: 'Probe Reranking',
                    data: [74.2, 68.2, 61.1],
                    backgroundColor: [
                        'rgba(46, 204, 113, 0.8)',
                        'rgba(46, 204, 113, 0.8)',
                        'rgba(231, 76, 60, 0.8)'
                    ],
                    borderColor: [
                        'rgba(46, 204, 113, 1)',
                        'rgba(46, 204, 113, 1)',
                        'rgba(231, 76, 60, 1)'
                    ],
                    borderWidth: 2,
                    borderRadius: 8,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + context.parsed.y.toFixed(1) + '%';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            },
            animation: {
                duration: 1500,
                easing: 'easeInOutQuart'
            }
        }
    });

    // Animate metrics on scroll
    const observerOptions = {
        threshold: 0.5,
        rootMargin: '0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateValue(entry.target);
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    document.querySelectorAll('.metric .value').forEach(element => {
        observer.observe(element);
    });

    function animateValue(element) {
        const finalValue = parseFloat(element.textContent);
        const duration = 1500;
        const startTime = performance.now();
        const startValue = 0;

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const easeProgress = 1 - Math.pow(1 - progress, 3); // Ease out cubic

            const currentValue = startValue + (finalValue - startValue) * easeProgress;
            element.textContent = currentValue.toFixed(3);

            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                element.textContent = finalValue.toFixed(3);
            }
        }

        requestAnimationFrame(update);
    }
});
