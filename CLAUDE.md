# Claude Code Spec-Driven Development

Kiro-style Spec Driven Development implementation using claude code slash commands, hooks and agents.

## Project Context

### Paths
- Steering: `.kiro/steering/`
- Specs: `.kiro/specs/`
- Commands: `.claude/commands/`

### Steering vs Specification

**Steering** (`.kiro/steering/`) - Guide AI with project-wide rules and context
**Specs** (`.kiro/specs/`) - Formalize development process for individual features

### Active Specifications
- **advanced-manifold-mathematics**: Comprehensive enhancement of RiemannAX with mathematical completeness improvements (Grassmann/SPD implementations), new manifolds (Product/Quotient), and Optimistix integration
- Check `.kiro/specs/` for active specifications
- Use `/kiro:spec-status [feature-name]` to check progress

## Development Guidelines
- Think in English, and generate responses in English for global accessibility

## Workflow

### Phase 0: Steering (Optional)
`/kiro:steering` - Create/update steering documents
`/kiro:steering-custom` - Create custom steering for specialized contexts

**Note**: Optional for new features or small additions. Can proceed directly to spec-init.

### Phase 1: Specification Creation
1. `/kiro:spec-init [detailed description]` - Initialize spec with detailed project description
2. `/kiro:spec-requirements [feature]` - Generate requirements document
3. `/kiro:spec-design [feature]` - Interactive: "Have you reviewed requirements.md? [y/N]"
4. `/kiro:spec-tasks [feature]` - Interactive: Confirms both requirements and design review

### Phase 2: Progress Tracking
`/kiro:spec-status [feature]` - Check current progress and phases

## Development Rules
1. **Consider steering**: Run `/kiro:steering` before major development (optional for new features)
2. **Follow 3-phase approval workflow**: Requirements → Design → Tasks → Implementation
3. **Approval required**: Each phase requires human review (interactive prompt or manual)
4. **No skipping phases**: Design requires approved requirements; Tasks require approved design
5. **Update task status**: Mark tasks as completed when working on them
6. **Keep steering current**: Run `/kiro:steering` after significant changes
7. **Check spec compliance**: Use `/kiro:spec-status` to verify alignment

## Python Development Constitution

### 📜 Absolute Quality Standards

**CONSTITUTIONAL LAW**: The following quality checks are **mandatory** for each Todo and **must ALL pass** before proceeding to the next Todo. **NO EXCEPTIONS ALLOWED**.

#### Mandatory Quality Checks (All Must Pass)
1. `mypy --config-file=pyproject.toml` - Complete type error resolution
2. `pre-commit run --all-files` - Complete style/quality issue resolution
3. `ruff check . --fix --unsafe-fixes` - Complete lint issue resolution
4. `pytest` - Complete test failure resolution

#### Information Quality Requirement
- **MANDATORY**: Use `context7` for precise, research-based implementations
- **PROHIBITED**: Implementations based on guesswork or general knowledge

#### Constitutional Enforcement
- ✅ **GREEN STATUS REQUIRED**: All 4 checks must show PASS status
- 🚫 **PROGRESSION BLOCKED**: Any RED/FAIL status blocks next Todo
- 🎯 **ZERO COMPROMISE**: Technical debt accumulation is constitutionally prohibited

#### Philosophy
This constitution eliminates technical debt at its source, ensuring every code change meets professional standards. Quality is non-negotiable.

## Steering Configuration

### Current Steering Files
Managed by `/kiro:steering` command. Updates here reflect command changes.

### Active Steering Files
- `product.md`: Always included - Product context and business objectives
- `tech.md`: Always included - Technology stack and architectural decisions
- `structure.md`: Always included - File organization and code patterns

### Custom Steering Files
<!-- Added by /kiro:steering-custom command -->
<!-- Format:
- `filename.md`: Mode - Pattern(s) - Description
  Mode: Always|Conditional|Manual
  Pattern: File patterns for Conditional mode
-->

- `python_development_constitution.md`: Conditional - *.py,pyproject.toml,requirements*.txt,setup.py,conftest.py,tests/**/*.py - Python Development Constitutional Law enforcing mandatory quality checks

### Inclusion Modes
- **Always**: Loaded in every interaction (default)
- **Conditional**: Loaded for specific file patterns (e.g., `"*.test.js"`)
- **Manual**: Reference with `@filename.md` syntax
