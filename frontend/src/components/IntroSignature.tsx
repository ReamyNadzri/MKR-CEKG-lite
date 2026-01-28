import React, { useState, useEffect, useRef } from 'react';

interface IntroSignatureProps {
    onComplete?: () => void;
    startDelay?: number;
    position?: { x: number; y: number };
    scale?: number;
    strokeWidth?: number;
}

const IntroSignature: React.FC<IntroSignatureProps> = ({
    onComplete,
    startDelay = 500,
    position = { x: 0, y: 0 },
    scale = 1,
    strokeWidth = 2,
}) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const [isReady, setIsReady] = useState(false);
    const [isVisible, setIsVisible] = useState(true);
    const [isFading, setIsFading] = useState(false);

    useEffect(() => {
        const loadAndAnimate = async () => {
            if (!containerRef.current) return;

            // Load SVG from public folder
            const response = await fetch('/signature.svg');
            const svgText = await response.text();

            // 1. Inject SVG into DOM
            containerRef.current.innerHTML = svgText;
            const svgEl = containerRef.current.querySelector('svg');

            if (!svgEl) {
                console.error("No SVG tag found in content");
                return;
            }

            // 2. Clean up dimensions for responsiveness
            svgEl.removeAttribute('width');
            svgEl.removeAttribute('height');
            svgEl.style.width = '100%';
            svgEl.style.height = 'auto';
            svgEl.style.overflow = 'visible';

            // 3. Find and Filter Paths
            const allPaths = Array.from(svgEl.querySelectorAll('path, line, polyline, polygon'));

            const validPaths = allPaths.filter(path => {
                if (path.closest('defs') || path.closest('clipPath')) return false;
                const stroke = path.getAttribute('stroke');
                const fill = path.getAttribute('fill');
                if ((!stroke || stroke === 'none') && (!fill || fill === 'none')) return false;
                return true;
            });

            const pathsToAnimate = validPaths.length > 0 ? validPaths : allPaths;

            // 4. Apply Calculation and Styles
            pathsToAnimate.forEach(path => {
                const svgPath = path as SVGPathElement;
                try {
                    const length = svgPath.getTotalLength();
                    svgPath.style.setProperty('--length', length.toString());
                } catch (e) {
                    svgPath.style.setProperty('--length', '1000');
                }

                svgPath.style.fill = 'transparent';
                svgPath.style.stroke = '#ffffff';
                svgPath.style.strokeWidth = 'var(--dynamic-stroke-width)';
                svgPath.style.strokeLinecap = 'round';
                svgPath.style.strokeLinejoin = 'round';
                svgPath.style.vectorEffect = 'non-scaling-stroke';
                svgPath.style.strokeDasharray = 'var(--length)';
                svgPath.style.strokeDashoffset = 'var(--length)';
            });

            setIsReady(true);

            // 5. Start Animation Sequence
            const startTimer = setTimeout(() => {
                pathsToAnimate.forEach(path => {
                    (path as SVGPathElement).style.animation = 'draw-exact 3.5s ease-out forwards';
                });

                const fadeTimer = setTimeout(() => setIsFading(true), 4000);
                const doneTimer = setTimeout(() => {
                    setIsVisible(false);
                    if (onComplete) onComplete();
                }, 5000);

                return () => {
                    clearTimeout(fadeTimer);
                    clearTimeout(doneTimer);
                };
            }, startDelay);

            return () => clearTimeout(startTimer);
        };

        loadAndAnimate();
    }, [startDelay, onComplete]);

    if (!isVisible) return null;

    return (
        <div
            className={`fixed inset-0 z-50 flex flex-col items-center justify-center text-white transition-all duration-1000 ${isFading ? 'opacity-0' : 'opacity-100'
                }`}
            style={{
                background: isFading
                    ? 'linear-gradient(to bottom right, #f8fafc, #faf5ff)'
                    : 'linear-gradient(to bottom right, #7c3aed, #a855f7, #c084fc)',
                ['--dynamic-stroke-width' as any]: `${strokeWidth}px`
            }}
        >
            <style>{`
        @keyframes draw-exact {
           from { stroke-dashoffset: var(--length); }
           to { stroke-dashoffset: 0; }
        }
      `}</style>

            <div className="relative w-full h-full flex items-center justify-center overflow-hidden">
                <div
                    className={`w-full h-full flex items-center justify-center transition-opacity duration-300 ${isReady ? 'opacity-100' : 'opacity-0'
                        }`}
                    style={{ transform: `translate(${position.x}px, ${position.y}px) scale(${scale})` }}
                >
                    <div
                        ref={containerRef}
                        className="w-full max-w-4xl [&>svg]:w-full [&>svg]:h-auto [&>svg]:overflow-visible"
                    />
                </div>
            </div>

            {!isFading && !isReady && (
                <p className="absolute bottom-10 text-gray-500 text-sm animate-pulse">Loading Signature...</p>
            )}
        </div>
    );
};

export default IntroSignature;
