# Archive

Code that is not a current product path. It is parked here so the history
stays in the tree. Nothing in this directory is installed, tested in CI, or
documented as part of Hyperfurion VK.

## `relay/`

An unpublished metered relay (Stripe, email login, a demo speech proxy, and
`api.hyperfurion.com`). It was never launched and is not active.

Do not deploy it as a backend for this project. Do not point the landing
page, the installers, or `voice-keyboard login` at it. Hyperfurion VK is
bring-your-own-key, or a local OpenAI-compatible server.

The tree is still readable if you want to study it. That is operator notes
for an unpublished service, not a user guide. The default test suite does
not collect these tests.

```bash
# optional, only if you are working on the archived relay
cd archive/relay
pip install .
python3 -m pytest tests -q
```
