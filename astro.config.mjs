// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	site: 'https://tlsdnwn55.github.io',
	base: '/space-notes',
	integrations: [
		starlight({
			title: 'Space Notes 🚀',
			description: '신우주(Space)의 마크다운 기반 기술 스터디 & 개발 문서',
			defaultLocale: 'root',
			locales: {
				root: {
					label: '한국어',
					lang: 'ko',
				},
			},
			customCss: ['./src/styles/custom.css'],
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/tlsdnwn55/space-notes' }
			],
			sidebar: [
				{
					label: '💻 Computer Science',
					items: [{ autogenerate: { directory: 'cs' } }],
				},
				{
					label: '🎨 Frontend',
					items: [{ autogenerate: { directory: 'frontend' } }],
				},
				{
					label: '⚙️ Backend',
					items: [{ autogenerate: { directory: 'backend' } }],
				},
				{
					label: '🤖 AI & Data',
					items: [{ autogenerate: { directory: 'ai' } }],
				},
			],
		}),
	],
});
