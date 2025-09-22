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
- `high-level-api-development` - Production-ready high-level API with scikit-learn style interface, practical problem templates, and ML library integration
- `advanced-manifold-mathematics` - Advanced mathematical foundations for manifold operations
- `hyperbolic-se3-manifolds` - Hyperbolic and SE(3) manifold implementations
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

- `constitution.md`: Conditional - *.py,pyproject.toml,requirements*.txt,setup.py,conftest.py,tests/**/*.py - Python Development Constitutional Law enforcing mandatory quality checks

### Inclusion Modes
- **Always**: Loaded in every interaction (default)
- **Conditional**: Loaded for specific file patterns (e.g., `"*.test.js"`)
- **Manual**: Reference with `@filename.md` syntax

## Commit Message & PR Title Standards

### Conventional Commits Format
All commit messages and PR titles MUST follow the Conventional Commits specification:

```
<type>[(scope)]: <description>

[optional body]

[optional footer(s)]
```

#### Required Components
- **type**: Categorizes the nature of change (REQUIRED)
- **description**: Brief summary in imperative mood, 50 chars max (REQUIRED)
- **scope**: Additional context, e.g., `(api)`, `(ui)`, `(docs)` (OPTIONAL)

#### Standard Types
- **feat**: New feature for users
- **fix**: Bug fix for users
- **docs**: Documentation changes
- **style**: Code formatting (no logic changes)
- **refactor**: Code restructuring (no behavior changes)
- **perf**: Performance improvements
- **test**: Test additions or modifications
- **chore**: Build/tooling changes, maintenance tasks
- **ci**: CI/CD configuration changes
- **build**: Build system or dependencies changes
- **revert**: Reverting previous commits

#### Breaking Changes
Use `!` after type/scope to indicate breaking changes:
- `feat!: remove deprecated API endpoint`
- `fix(auth)!: change token validation logic`

#### Message Quality Rules
1. **Imperative mood**: "Add feature" not "Added feature"
2. **Lowercase first letter**: Start the description with a lowercase letter.
3. **No period**: Don't end description with `.`
4. **50 char limit**: Keep description concise and scannable
5. **Present tense**: Describe what the commit does, not what it did

#### Body Guidelines (Optional)
- Separate from description with blank line
- Wrap at 72 characters
- Explain **what** and **why**, not **how**
- Include breaking change details
- Reference issues: `Fixes #123`, `Closes #456`, `Refs #789`

#### Footer Examples (Optional)
```
BREAKING CHANGE: API endpoint /users now requires authentication
Reviewed-by: John Doe
Refs: #123, #456
Co-authored-by: Jane Smith <jane@example.com>
```

### Examples

#### Good Examples
```bash
# Simple feature
feat: add dark mode toggle to settings

# Feature with scope
feat(api): add user authentication middleware

# Bug fix with issue reference
fix: resolve memory leak in image processing

Fixes race condition when processing large batches of images
that caused memory usage to grow indefinitely.

Closes #234

# Breaking change
feat!: upgrade to Node.js 18

BREAKING CHANGE: Node.js 16 support removed. Minimum version is now 18.0.0

# Chore with scope
chore(deps): update JAX to 0.4.25
```

#### Bad Examples
```bash
# Too vague
fix: bugs

# Wrong tense
feat: added new feature

# Too long
feat: implement the new authentication system with JWT tokens and refresh token rotation

# Wrong capitalization
Fix: Authentication issue

# Unnecessary period
docs: update README.md.
```

### PR Title Standards

#### PR Title = Squash Merge Title
- PR titles become commit messages when using "Squash & Merge"
- MUST follow conventional commit format
- Configure GitHub to "Default to PR title for squash merge commits"

#### PR-Specific Guidelines
- **Single line only**: No body in PR title (use PR description instead)
- **Breaking changes**: Use `!` notation since multi-line not available
- **Issue linking**: Use PR description, not title
- **Draft PRs**: Prefix with `[WIP]` or use GitHub Draft feature

#### Automation Integration
- **Semantic Release**: Enables automatic version bumping and changelog generation
- **Auto-labeling**: GitHub Actions can auto-label PRs based on type
- **Validation**: Use `action-semantic-pull-request` to enforce standards

### Team Workflow Integration

#### Development Process
1. **Branch naming**: Use conventional format `feat/add-auth`, `fix/memory-leak`
2. **Commit early, commit often**: Individual commits can be informal during development
3. **PR review**: Ensure PR title follows conventional format before merge
4. **Squash strategy**: Use squash merging to create clean history with conventional titles

#### Quality Gates
- PR title validation via GitHub Actions
- Commit message linting in pre-commit hooks
- Automated changelog generation from conventional commits
- Semantic versioning based on commit types

### Configuration Files
- **commitlint**: Enforce conventional commits locally
- **semantic-release**: Automate releases based on commit messages
- **GitHub Actions**: Validate PR titles and automate labeling
