import { useEffect, useRef } from 'react';
import './AnimatedBackground.css';

const AnimatedBackground = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        // Check for reduced motion preference
        const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

        if (prefersReducedMotion) {
            return;
        }

        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Set canvas size
        const resizeCanvas = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        };
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        // Gradient blob configuration
        const blobs: Array<{
            x: number;
            y: number;
            radius: number;
            vx: number;
            vy: number;
            color: string;
        }> = [];

        const colors = [
            'rgba(124, 58, 237, 0.15)',   // Purple
            'rgba(192, 38, 211, 0.12)',   // Fuchsia
            'rgba(236, 72, 153, 0.1)',    // Pink
            'rgba(99, 102, 241, 0.13)',   // Indigo
            'rgba(59, 130, 246, 0.1)',    // Blue
        ];

        // Create blobs
        for (let i = 0; i < 5; i++) {
            blobs.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                radius: Math.random() * 300 + 200,
                vx: (Math.random() - 0.5) * 0.3,
                vy: (Math.random() - 0.5) * 0.3,
                color: colors[i],
            });
        }

        // Animation loop
        let animationFrameId: number;

        const animate = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw gradient background
            const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
            gradient.addColorStop(0, '#f8fafc');
            gradient.addColorStop(1, '#f1f5f9');
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Update and draw blobs
            blobs.forEach((blob) => {
                blob.x += blob.vx;
                blob.y += blob.vy;

                // Bounce off edges
                if (blob.x < 0 || blob.x > canvas.width) blob.vx *= -1;
                if (blob.y < 0 || blob.y > canvas.height) blob.vy *= -1;

                // Draw blob with radial gradient
                const blobGradient = ctx.createRadialGradient(blob.x, blob.y, 0, blob.x, blob.y, blob.radius);
                blobGradient.addColorStop(0, blob.color);
                blobGradient.addColorStop(1, 'rgba(255, 255, 255, 0)');

                ctx.fillStyle = blobGradient;
                ctx.beginPath();
                ctx.arc(blob.x, blob.y, blob.radius, 0, Math.PI * 2);
                ctx.fill();
            });

            animationFrameId = requestAnimationFrame(animate);
        };

        animate();

        return () => {
            window.removeEventListener('resize', resizeCanvas);
            cancelAnimationFrame(animationFrameId);
        };
    }, []);

    return (
        <canvas
            ref={canvasRef}
            className="animated-background"
            aria-hidden="true"
        />
    );
};

export default AnimatedBackground;
