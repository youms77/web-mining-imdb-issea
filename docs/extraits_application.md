# Textes du corpus affichés par l'application

Fichier produit par `src/app_artefacts.py`. **Licence du corpus : [à vérifier].** Décision de publication à prendre en phase 8, avant tout dépôt public.

Relecture faite en phase 7 : pas d'adresse, de courriel, de lien ni d'information sur la vie privée d'un auteur. Les extraits sont en minuscules, car ils viennent du texte nettoyé.

## Critique entière (exemple « mitigé »)

- `id_brut` 13669, partie entraînement, 80 mots.
- Texte : To summerize this movie: nice for TV but too small for the theatre. I enjoyed watching this movie at home but I wont watch it a second time. The concept is good, but what ends up in the movie is just a summary. The end had a 'nice' twist but is still unsatisfying. Maybe it was the intention of the director but it wasn't worked out like a it should be. But then again, it's an OK kind of movie.

## Extraits de thèmes NMF (8, 160 caractères au plus hors points de suspension)

| Thème | `id_brut` | Extrait |
|---|---|---|
| Récits de vie et de guerre | 27646 | … documentaries. this one tells of two warring factions in colombia, guerrillas and paramilitaries, and the surrounding peoples of medellin. guns, drugs and … |
| Déception, film raté | 39679 | … would have given this a 0, if possible--bad acting, bad directing, bad production, bad plot. this was made in 2001 and it looks more like 1965. very low … |
| Comédie et humour | 7586 | … the harassing cops to their undies and handcuff them in the park in a unflattering position. beyond that, the jokes are not funny, the characters are not … |
| Personnages et interprétation | 4201 | … that is totally paranoic and unbelievable. but city hall does not do neither of the above mentioned. the plot is cleverly crafted, story is believable. as far … |
| Séries télévisées | 46199 | … i got this as a complete set of 9 episodes on 5 dvds. i knew nothing about the history of the series. season 1 of the series has the pilot episode as episode 1 … |
| Coups de cœur, famille, musique | 21309 | … love. gorgeous love. dumb love. love love love. minnie moore understands that what she really needs is a man who trust her, trust her and love her madly. of … |
| Horreur | 12719 | … i consider myself a huge horror movie fan. one night i wanted to just rent some newer horror movies to make fun of. then, i rented milo. i was so surprised by … |
| Adaptations de livres | 16137 | … i love memoirs of a geisha so i read the book twice; it is one of the best book i have read last year. i was looking forward to the movie and was afraid that … |

## Si la publication de ces textes est refusée

Vider `extraits` dans `models/deploy/contenu_app.json` et remplacer l'exemple mitigé par un texte rédigé. L'application masque alors les extraits ; le reste fonctionne sans eux.
