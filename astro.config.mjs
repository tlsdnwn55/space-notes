// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { visit } from 'unist-util-visit';

/**
 * Standard Rehype plugin to convert mermaid code blocks to <div class="mermaid">
 */
function rehypeMermaid() {
	return (tree) => {
		visit(tree, 'element', (node, index, parent) => {
			if (
				node.tagName === 'pre' &&
				node.children &&
				node.children.length > 0 &&
				node.children[0].tagName === 'code'
			) {
				const codeNode = node.children[0];
				const className = codeNode.properties?.className || [];
				const isMermaid = Array.isArray(className) && className.includes('language-mermaid');
				if (isMermaid) {
					const text = codeNode.children.map((c) => c.value || '').join('');
					parent.children[index] = {
						type: 'element',
						tagName: 'div',
						properties: { className: ['mermaid'] },
						children: [{ type: 'text', value: text }],
					};
				}
			}
		});
	};
}

// https://astro.build/config
export default defineConfig({
	site: 'https://tlsdnwn55.github.io',
	base: '/space-notes',
	markdown: {
		remarkPlugins: [remarkMath],
		rehypePlugins: [rehypeMermaid, rehypeKatex],
	},
	integrations: [
		starlight({
			title: 'space-notes',
			description: 'Personal Technical Journal & Engineering Notes',
			defaultLocale: 'root',
			locales: {
				root: {
					label: '한국어',
					lang: 'ko',
				},
			},
			customCss: ['./src/styles/custom.css'],
			head: [
				{
					tag: 'link',
					attrs: {
						rel: 'stylesheet',
						href: 'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.11/katex.min.css',
					},
				},
				{
					tag: 'script',
					attrs: {
						src: 'https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.9.1/mermaid.min.js',
					},
				},
				{
					tag: 'script',
					content: `
						function initAppScripts() {
							// 1. Sidebar Toggle Button
							const titleWrapper = document.querySelector('.title-wrapper');
							if (titleWrapper && !document.querySelector('.sidebar-toggle-btn')) {
								const toggleBtn = document.createElement('button');
								toggleBtn.className = 'sidebar-toggle-btn';
								toggleBtn.type = 'button';
								toggleBtn.setAttribute('aria-label', 'Toggle Sidebar');
								toggleBtn.innerHTML = '☰';
								toggleBtn.title = '사이드바 접기/열기';
								toggleBtn.addEventListener('click', () => {
									const isCollapsed = document.documentElement.dataset.sidebarCollapsed === 'true';
									document.documentElement.dataset.sidebarCollapsed = isCollapsed ? 'false' : 'true';
								});
								titleWrapper.prepend(toggleBtn);
							}

							// 2. Mermaid Diagram Renderer (High-Contrast Clean Theme)
							if (window.mermaid) {
								const isDark = document.documentElement.dataset.theme !== 'light';
								window.mermaid.initialize({
									startOnLoad: false,
									theme: 'base',
									themeVariables: {
										darkMode: isDark,
										background: 'transparent',
										primaryColor: isDark ? '#1e293b' : '#f8fafc',
										primaryTextColor: isDark ? '#ffffff' : '#0f172a',
										primaryBorderColor: '#38bdf8',
										lineColor: '#38bdf8',
										nodeBorder: '#38bdf8',
										clusterBkg: isDark ? '#0f172a' : '#f1f5f9',
										clusterBorder: '#64748b',
										fontSize: '15px',
										fontFamily: 'Inter, sans-serif'
									}
								});
								window.mermaid.run({ querySelector: '.mermaid' });
							}

						}

						if (document.readyState === 'complete' || document.readyState === 'interactive') {
							setTimeout(initAppScripts, 150);
						} else {
							document.addEventListener('DOMContentLoaded', () => setTimeout(initAppScripts, 150));
						}
						document.addEventListener('astro:page-load', () => setTimeout(initAppScripts, 150));
					`,
				},
			],
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/tlsdnwn55/space-notes' }
			],
			sidebar: [
				{
					label: 'All Articles',
					items: [{ autogenerate: { directory: 'posts' } }],
				},
			],
		}),
	],
});
