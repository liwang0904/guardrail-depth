# Can CLIP see each artist's style?

With no guardrail, an artist counts as recognizable when CLIP names them more often for their own prompts than for other artists' prompts.

| artist             | group              |   hit |   false_label |   lift |
|:-------------------|:-------------------|------:|--------------:|-------:|
| M.C. Escher        | deceased_copyright | 0.000 |         0.000 |  0.000 |
| Salvador Dali      | deceased_copyright | 0.000 |         0.000 |  0.000 |
| Thomas Kinkade     | deceased_copyright | 0.000 |         0.000 |  0.000 |
| Pablo Picasso      | deceased_copyright | 0.250 |         0.010 |  0.240 |
| Andy Warhol        | deceased_copyright | 0.875 |         0.005 |  0.870 |
| Greg Rutkowski     | living             | 0.000 |         0.052 | -0.052 |
| Loish              | living             | 0.000 |         0.010 | -0.010 |
| Kelly McKernan     | living             | 0.000 |         0.005 | -0.005 |
| Simon Stalenhag    | living             | 0.000 |         0.005 | -0.005 |
| Sarah Andersen     | living             | 0.000 |         0.000 |  0.000 |
| Tyler Edlin        | living             | 0.250 |         0.130 |  0.120 |
| Artgerm            | living             | 0.125 |         0.000 |  0.125 |
| Hollie Mengert     | living             | 0.375 |         0.062 |  0.312 |
| Kilian Eng         | living             | 0.625 |         0.068 |  0.557 |
| Karla Ortiz        | living             | 0.875 |         0.234 |  0.641 |
| Albrecht Durer     | public_domain      | 0.125 |         0.000 |  0.125 |
| Francisco Goya     | public_domain      | 0.125 |         0.000 |  0.125 |
| Claude Monet       | public_domain      | 0.250 |         0.000 |  0.250 |
| Rembrandt          | public_domain      | 0.250 |         0.000 |  0.250 |
| Edvard Munch       | public_domain      | 0.375 |         0.000 |  0.375 |
| Paul Cezanne       | public_domain      | 0.375 |         0.000 |  0.375 |
| Georges Seurat     | public_domain      | 0.500 |         0.000 |  0.500 |
| Katsushika Hokusai | public_domain      | 0.750 |         0.000 |  0.750 |
| Vincent van Gogh   | public_domain      | 0.750 |         0.000 |  0.750 |
| Gustav Klimt       | public_domain      | 0.875 |         0.000 |  0.875 |

Not recognizable above chance: 8 of 25 artists (Greg Rutkowski, Kelly McKernan, Loish, M.C. Escher, Salvador Dali, Sarah Andersen, Simon Stalenhag, Thomas Kinkade). Style protection cannot be measured for them with CLIP.
