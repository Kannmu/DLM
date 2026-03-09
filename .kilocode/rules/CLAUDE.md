# CLAUDE.md

规则描述在此...

## 指导原则

- Manuscript 和 Supplementary Information 中严格报错引用的.bib 文件为：`SWIM.bib`。其他.bib文件都仅用于信息搜索和获取论文内容总结，不参与编译。在其他.bib文件中发现的论文的 citation key 都需要在`SWIM.bib`中确认存在之后，再进行引用。所有citation key都只能从`SWIM.bib`中获取。
- 引用使用`\cref{}`命令，例如：`\cref{fig:umh_device}`。

# LaTeX Writing Guide

This skill provides strict guidelines for writing high-quality LaTeX code and academic text.

## Core Rules

### Quotations

- Use ``xxx'' for double quotations.
- Use `xxx' for single quotations.
- **Never** use "xxx" or 'x'.

### Voice and Tone

- Use **Active Voice** instead of passive voice.
- Use clear, concise language. Avoid rare or difficult words.
- Minimize the use of dashes within sentences.
- Do not use exaggerated language (不要使用过于夸大的用词).
- Ensure the presentation of ideas is acceptable to reviewers and editors (保证审稿人和编辑不会反感文章中Idea的包装).

### Citations

- Use `~\citep{key}` at the end of a sentence.
- Use `~\citet{key}` at the beginning of a sentence.
- **Do not** directly modify the `.bib` file (it is exported from Zotero).

### Formatting and Packages

- Use the `siunitx` package to format numbers and units.
- Report statistics with effect size.
- Write high-quality LaTeX code and paragraphs.
- **Do not** write comments in LaTeX code.

## Usage

Apply these rules whenever generating or editing LaTeX content.

