"""Educational species reference data for the BlueEye Species Explorer.

This is a **reference library**, not a claim about what BlueEye can detect.
Every entry carries an explicit ``detection`` capability so the UI never
implies that a generic ``fish`` / ``turtle`` / ``shark`` model identifies a
species:

* ``direct``      - an active model has a class for this exact animal
* ``broad``       - only a generic group class exists (e.g. ``fish``); the
                    species identity is **not** established by the model
* ``none``        - no active model has any applicable class
* ``unverified``  - not enough evidence to make a claim

Each profile is a set of educational fields (overview, identification, habitat
and distribution, diet and behaviour, ecological importance, conservation and
facts). Biological facts are drawn from public conservation and research
sources; the conservation status names its source, and every profile carries
``references`` to authoritative sources that were checked as reachable.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

#: Detection-capability categories -> human label (single source of truth).
DETECTION_CATEGORIES: dict[str, str] = {
    "direct": "Direct class support",
    "broad": "Broad-category detection only",
    "none": "Not currently supported",
    "unverified": "Unverified",
}

#: Order used when listing the capability legend.
DETECTION_ORDER: tuple[str, ...] = ("direct", "broad", "none", "unverified")

#: Conservation-status source (text citation; not a URL).
_IUCN = (
    "IUCN Red List of Threatened Species - global assessment "
    "(most recent published assessment available at the time of writing)"
)
_NE = "Not assessed on the IUCN Red List of Threatened Species"


@dataclass(frozen=True)
class Species:
    """One aquatic species profile (educational reference data)."""

    id: str
    common_name: str
    scientific_name: str
    group: str
    habitat: str
    overview: str
    identification: str
    habitat_notes: str
    distribution: str
    diet: str
    behaviour: str
    ecological_role: str
    threats: str
    conservation_status: str
    conservation_source: str
    conservation_actions: str
    facts: tuple[str, ...]
    detection: str
    detection_note: str
    references: tuple[tuple[str, str], ...]
    image: str = ""
    image_credit: str = ""

    @property
    def detection_label(self) -> str:
        return DETECTION_CATEGORIES.get(self.detection, "Unverified")


def _sp(**kwargs) -> Species:
    return Species(**kwargs)


# --------------------------------------------------------------------------- #
# Verified reference builders (every URL below was checked to resolve)
# --------------------------------------------------------------------------- #
def _fishbase(scientific_name: str) -> tuple[str, str]:
    slug = scientific_name.replace(" ", "-")
    return ("FishBase (species summary)", f"https://www.fishbase.se/summary/{slug}.html")


def _sealifebase(scientific_name: str) -> tuple[str, str]:
    slug = scientific_name.replace(" ", "-")
    return ("SeaLifeBase (species summary)", f"https://www.sealifebase.se/summary/{slug}.html")


def _reptile_database(scientific_name: str) -> tuple[str, str]:
    genus, species = scientific_name.split(" ", 1)
    url = f"https://reptile-database.reptarium.cz/species?genus={genus}&species={species}"
    return ("The Reptile Database", url)


def _adw(scientific_name: str) -> tuple[str, str]:
    slug = scientific_name.replace(" ", "_")
    return ("Animal Diversity Web", f"https://animaldiversity.org/accounts/{slug}/")


def _cms(scientific_name: str) -> tuple[str, str]:
    slug = scientific_name.lower().replace(" ", "-")
    return (
        "Convention on Migratory Species (UNEP/CMS)",
        f"https://www.cms.int/en/species/{slug}",
    )


def _fish_refs(scientific_name: str) -> tuple[tuple[str, str], ...]:
    return (_fishbase(scientific_name), _adw(scientific_name))


def _invert_refs(scientific_name: str) -> tuple[tuple[str, str], ...]:
    return (_sealifebase(scientific_name), _adw(scientific_name))


def _reptile_refs(scientific_name: str) -> tuple[tuple[str, str], ...]:
    return (_reptile_database(scientific_name), _adw(scientific_name))


def _mammal_refs(scientific_name: str) -> tuple[tuple[str, str], ...]:
    return (_adw(scientific_name), _cms(scientific_name))


#: The full reference library. Indian freshwater and South Asian species come
#: first, then Indian coastal/marine life.
SPECIES: tuple[Species, ...] = (
    # ------------------------------------------------------------------ #
    # Indian freshwater fish
    # ------------------------------------------------------------------ #
    _sp(
        id="rohu",
        common_name="Rohu",
        scientific_name="Labeo rohita",
        group="Freshwater fish",
        habitat="Freshwater",
        overview=(
            "The rohu is a large, silvery carp and one of the most important "
            "freshwater food fishes of the Indian subcontinent. It is a mid-water "
            "grazer found in rivers, floodplains and tanks, and a cornerstone of "
            "South Asian aquaculture."
        ),
        identification=(
            "A large cyprinid reaching roughly 1-2 m, with reddish fins, a blunt "
            "rounded head, large silvery scales and a single pair of barbels."
        ),
        habitat_notes=(
            "Slow-flowing rivers, floodplains, tanks and reservoirs; tolerant of a "
            "wide temperature range and feeds in the water column."
        ),
        distribution=(
            "Native to the Ganga-Brahmaputra and peninsular river systems of the "
            "Indian subcontinent; widely introduced and farmed across South and "
            "South-East Asia for aquaculture."
        ),
        diet="Mostly herbivorous: phytoplankton, algae, submerged vegetation and organic detritus.",
        behaviour=(
            "A shoaling mid-water grazer that feeds mainly by day, and migrates "
            "upstream to breed in flooded rivers during the monsoon."
        ),
        ecological_role=(
            "Converts plant matter into fish biomass and links primary production "
            "to larger predators; a key food fish and pillar of polyculture farming."
        ),
        threats=(
            "Habitat alteration, dams and river fragmentation, water abstraction "
            "and pollution; wild stocks also face heavy fishing pressure."
        ),
        conservation_status="Least Concern",
        conservation_source=_IUCN,
        conservation_actions=(
            "Extensive aquaculture supplies most of the market and reduces pressure "
            "on wild stocks; river connectivity and water-quality protection help "
            "wild populations."
        ),
        facts=(
            "Along with catla and mrigal it forms the 'Indian major carp' trio that underpins freshwater aquaculture in the region.",
            "It is one of the most widely farmed freshwater fishes in the world.",
        ),
        detection="broad",
        detection_note=(
            "Only a generic 'fish' class exists (in the brackish / underwater / "
            "aquarium models). None is trained on Indian freshwater species, so the "
            "model cannot confirm this is Rohu."
        ),
        references=_fish_refs("Labeo rohita"),
    ),
    _sp(
        id="catla",
        common_name="Catla",
        scientific_name="Labeo catla",
        group="Freshwater fish",
        habitat="Freshwater",
        overview=(
            "Catla is a large, deep-bodied major carp and the fastest-growing of "
            "the Indian major carps. A surface filter-feeder, it is native to "
            "northern Indian rivers and is now farmed throughout South Asia."
        ),
        identification=(
            "A deep body with a large head, a prominent lower jaw and an upturned "
            "mouth; silvery, and can grow to about 1.8 m."
        ),
        habitat_notes=(
            "Rivers and floodplain waters, feeding near the surface; also farmed in "
            "ponds and reservoirs."
        ),
        distribution=(
            "Native to the Ganga, Brahmaputra and Mahanadi river systems of the "
            "Indian subcontinent; introduced widely for aquaculture."
        ),
        diet="Zooplankton and phytoplankton filter-feeder, feeding near the water surface.",
        behaviour=(
            "Shoals in surface waters and feeds by filtering plankton; breeds in "
            "rivers during the monsoon."
        ),
        ecological_role=(
            "Surface planktivore that harvests plankton production; a staple "
            "aquaculture and capture-fishery species."
        ),
        threats="River fragmentation, flow regulation and pollution; aquaculture now supplies most of the market.",
        conservation_status="Least Concern",
        conservation_source=_IUCN,
        conservation_actions=(
            "Sustained by extensive aquaculture; wild populations benefit from "
            "river connectivity and water-quality protection."
        ),
        facts=(
            "It was previously placed in its own genus, Catla, and is now classified as Labeo catla.",
            "Its upturned mouth is specialised for straining plankton from surface water.",
        ),
        detection="broad",
        detection_note="Falls under a generic 'fish' class only; no active model is specific to Indian carps.",
        references=_fish_refs("Labeo catla"),
    ),
    _sp(
        id="mrigal",
        common_name="Mrigal",
        scientific_name="Cirrhinus mrigala",
        group="Freshwater fish",
        habitat="Freshwater",
        overview=(
            "Mrigal is a slender, bottom-feeding major carp of the Indian "
            "subcontinent and the third of the 'Indian major carp' triad. It grubs "
            "the bottom for detritus and is extensively farmed."
        ),
        identification=(
            "A slender greyish-silver carp with a pointed head and a single short "
            "barbel; typically 30-100 cm."
        ),
        habitat_notes="Rivers, ponds and tanks, feeding on or near the bottom.",
        distribution=(
            "Indian subcontinent rivers, especially the Ganga and peninsular "
            "systems; farmed throughout the region."
        ),
        diet="Bottom-dwelling detritivore feeding on decaying plant matter, algae and mud-dwelling organisms.",
        behaviour="A benthic feeder that stirs the substrate; breeds in flooded rivers during the monsoon.",
        ecological_role="Benthic detritivore that recycles organic matter and helps keep pond bottoms clean in polyculture.",
        threats="Habitat degradation, pollution and barriers to migration; heavily relied upon as a farmed species.",
        conservation_status="Least Concern",
        conservation_source=_IUCN,
        conservation_actions="Maintained by aquaculture; wild stocks depend on river health and seasonal flooding.",
        facts=(
            "Its habit of grubbing the pond bottom keeps organic waste in check in polyculture ponds.",
        ),
        detection="broad",
        detection_note="Covered only by a generic 'fish' class; no Indian-specific model.",
        references=_fish_refs("Cirrhinus mrigala"),
    ),
    _sp(
        id="hilsa",
        common_name="Hilsa (Indian shad)",
        scientific_name="Tenualosa ilisha",
        group="Freshwater fish",
        habitat="Freshwater",
        overview=(
            "Hilsa, or Indian shad, is a celebrated anadromous herring of the "
            "Ganga-Brahmaputra delta and the northern Indian Ocean. It migrates up "
            "rivers to spawn and supports one of South Asia's most important fisheries."
        ),
        identification=(
            "A laterally compressed, silver herring-like fish with a saw-toothed "
            "belly keel; usually 30-50 cm."
        ),
        habitat_notes=(
            "Estuaries, coastal waters and large rivers; adults migrate from the "
            "sea into fresh water to breed."
        ),
        distribution=(
            "Ganga-Brahmaputra deltas and coastal waters of the northern Indian "
            "Ocean; famous from the Hooghly and Padma, and the national fish of "
            "Bangladesh."
        ),
        diet="Plankton-feeder as an adult, feeding in estuarine and coastal waters.",
        behaviour=(
            "Anadromous: matures at sea and runs up rivers to spawn, with "
            "well-documented monsoon-season migrations."
        ),
        ecological_role="Links marine and freshwater food webs and supports a culturally and economically vital fishery.",
        threats=(
            "Overfishing (especially of juveniles), dams and barrages that block "
            "migration routes, and reduced dry-season flow."
        ),
        conservation_status="Least Concern",
        conservation_source=_IUCN,
        conservation_actions=(
            "Fishery closures during peak breeding, minimum-size rules and habitat "
            "protection are used in India and Bangladesh."
        ),
        facts=(
            "Hilsa runs are celebrated in Bengal and Bangladesh; the fish is the national fish of Bangladesh.",
        ),
        detection="broad",
        detection_note="Only a generic 'fish' class is available; species identity is not established.",
        references=_fish_refs("Tenualosa ilisha"),
    ),
    _sp(
        id="golden_mahseer",
        common_name="Golden mahseer",
        scientific_name="Tor putitora",
        group="Freshwater fish",
        habitat="Freshwater",
        overview=(
            "The golden mahseer is a large, powerful cyprinid of Himalayan foothill "
            "rivers and one of the world's largest freshwater carps. It is a "
            "flagship of Himalayan river conservation and a prized game fish."
        ),
        identification=(
            "A golden-olive body with large scales and broad fins; can exceed 2 m "
            "and 50 kg."
        ),
        habitat_notes=(
            "Clear, fast-flowing rivers and streams with rocky or gravelly beds; "
            "also enters reservoirs."
        ),
        distribution=(
            "Foothill rivers of the Himalaya and northern India (Ganga, Brahmaputra "
            "and Indus drainages); populations have declined across much of the range."
        ),
        diet="Omnivorous: small fish, crustaceans, insects and fruit fallen into rivers.",
        behaviour="A strong swimmer that ascends rivers and tributaries to spawn; long-lived and slow-growing.",
        ecological_role="Top freshwater predator that helps structure river fish communities; a conservation flagship.",
        threats="Dams and barrages, habitat degradation, pollution, poaching and unregulated fishing.",
        conservation_status="Endangered",
        conservation_source=_IUCN,
        conservation_actions=(
            "Protected under Indian wildlife law; captive breeding and ranching "
            "programmes, angling bans and river protection support recovery."
        ),
        facts=(
            "One of the largest freshwater carps in the world and a beloved 'tiger of the river' for anglers.",
        ),
        detection="broad",
        detection_note="Only a generic 'fish' class exists; no model is trained on mahseer.",
        references=_fish_refs("Tor putitora"),
    ),
    _sp(
        id="chocolate_mahseer",
        common_name="Chocolate mahseer",
        scientific_name="Neolissochilus hexagonolepis",
        group="Freshwater fish",
        habitat="Freshwater",
        overview=(
            "The chocolate mahseer is a medium-sized hill-stream carp of the eastern "
            "Himalaya and North-East India. It is important to local food security "
            "and river angling."
        ),
        identification="A deep-bodied olive-brown fish with large scales; reaches about 60 cm.",
        habitat_notes="Clear, cool hill streams and rivers with rocky beds.",
        distribution="Hill streams of the eastern Himalaya and North-East India (Brahmaputra tributaries).",
        diet="Omnivorous drift-feeder on insects, algae and small invertebrates.",
        behaviour="Feeds on drifting material in flowing water and moves within stream systems seasonally.",
        ecological_role="Stream fish that links aquatic insect production to larger predators; a local food and sport fish.",
        threats="Hydropower development, stream siltation, destructive fishing and habitat loss.",
        conservation_status="Near Threatened",
        conservation_source=_IUCN,
        conservation_actions=(
            "Included in regional conservation and aquaculture efforts; protection "
            "depends on maintaining free-flowing streams."
        ),
        facts=(
            "In Meghalaya and Arunachal Pradesh it is a traditional food fish caught by indigenous river communities.",
        ),
        detection="broad",
        detection_note="Only a generic 'fish' class exists.",
        references=_fish_refs("Neolissochilus hexagonolepis"),
    ),
    _sp(
        id="deccan_mahseer",
        common_name="Deccan mahseer (blue-finned mahseer)",
        scientific_name="Tor khudree",
        group="Freshwater fish",
        habitat="Freshwater",
        overview=(
            "The Deccan (blue-finned) mahseer is a powerful river carp of peninsular "
            "India's Western Ghats. It is an apex river predator and a focus of "
            "catch-and-release sport fishing."
        ),
        identification="A stocky, bluish-grey mahseer with large scales, reaching about 1.5 m.",
        habitat_notes="Fast-flowing, rocky rivers and reservoirs of the Western Ghats.",
        distribution=(
            "Rivers of peninsular India, especially the Western Ghats; native to the "
            "Cauvery, Krishna and Godavari systems."
        ),
        diet="Omnivorous; feeds on fish, insects, molluscs and fruits.",
        behaviour="Ascends rivers to spawn before the monsoon; long-lived and territorial in pools.",
        ecological_role="Apex predator in Ghat streams; important for ecotourism, sport fishing and seed dispersal of riparian fruit.",
        threats="Dams and water diversion, habitat loss, pollution and illegal fishing.",
        conservation_status="Endangered",
        conservation_source=_IUCN,
        conservation_actions=(
            "Released by anglers after capture in catch-and-release fisheries; "
            "conservation breeding and river protection are used in parts of its range."
        ),
        facts=(
            "It supports a catch-and-release sport fishery in the Cauvery.",
            "It is one of several mahseer species, a group prized by anglers as 'tigers of the river'.",
        ),
        detection="broad",
        detection_note="Only a generic 'fish' class exists; no species-specific model.",
        references=_fish_refs("Tor khudree"),
    ),
    _sp(
        id="striped_snakehead",
        common_name="Striped snakehead (murrel)",
        scientific_name="Channa striata",
        group="Freshwater fish",
        habitat="Freshwater",
        overview=(
            "The striped snakehead, or murrel, is an elongated predatory fish "
            "widespread across South and South-East Asia. It can breathe air and "
            "survive in poorly oxygenated water."
        ),
        identification=(
            "An elongated, snake-like grey-brown body with darker bands and long "
            "dorsal and anal fins."
        ),
        habitat_notes=(
            "Ponds, wetlands, canals and slow rivers, including oxygen-poor waters; "
            "can gulp air at the surface."
        ),
        distribution=(
            "Widespread across South and South-East Asia, including most Indian "
            "freshwater systems; introduced in some areas outside its native range."
        ),
        diet="Carnivorous ambush predator of fish, frogs, insects and crustaceans.",
        behaviour=(
            "A solitary ambush hunter, often among vegetation; tolerant of low oxygen "
            "and able to move short distances overland in wet conditions."
        ),
        ecological_role="Predator that structures prey communities in ponds and wetlands; widely cultured and valued as food.",
        threats="Generally resilient; local pressures include wetland drainage and pollution.",
        conservation_status="Least Concern",
        conservation_source=_IUCN,
        conservation_actions="Widely farmed, which supplies much of the demand; wetland protection benefits wild populations.",
        facts=(
            "It can survive in oxygen-poor water by gulping air at the surface.",
        ),
        detection="broad",
        detection_note="Only a generic 'fish' class exists.",
        references=_fish_refs("Channa striata"),
    ),
    _sp(
        id="indian_featherback",
        common_name="Indian featherback (clown knifefish)",
        scientific_name="Chitala chitala",
        group="Freshwater fish",
        habitat="Freshwater",
        overview=(
            "The Indian featherback, or clown knifefish, is a deep-bodied predatory "
            "fish of Indian rivers and floodplains, named for the row of dark spots "
            "along its flanks."
        ),
        identification=(
            "A knife-shaped silvery body with a row of dark spots and a very long "
            "anal fin; can reach about 1 m."
        ),
        habitat_notes="Slow rivers, floodplains and tanks, often near submerged cover.",
        distribution="Rivers and floodplains of the Indian subcontinent, including the Ganga and Brahmaputra basins.",
        diet="Carnivorous; feeds on fish and invertebrates, hunting at dusk and night.",
        behaviour="A nocturnal predator that shelters by day and breeds in flooded vegetation during the monsoon.",
        ecological_role="Predatory fish of slow rivers and floodplains; important in local fisheries and increasingly in the ornamental trade.",
        threats="Overfishing, habitat loss and the aquarium trade.",
        conservation_status="Near Threatened",
        conservation_source=_IUCN,
        conservation_actions="Breeding and ranching efforts exist; floodplain protection supports wild populations.",
        facts=(
            "Despite its 'knifefish' name it is not related to the South American knifefishes.",
        ),
        detection="broad",
        detection_note="Only a generic 'fish' class exists.",
        references=_fish_refs("Chitala chitala"),
    ),
    # ------------------------------------------------------------------ #
    # South Asian freshwater turtles
    # ------------------------------------------------------------------ #
    _sp(
        id="indian_softshell_turtle",
        common_name="Indian softshell turtle (Ganges softshell)",
        scientific_name="Nilssonia gangetica",
        group="Freshwater turtle",
        habitat="Freshwater",
        overview=(
            "The Indian softshell turtle, or Ganges softshell, is a large freshwater "
            "turtle with a flexible leathery shell. It is found in major northern "
            "Indian river systems and is sacred in parts of India."
        ),
        identification=(
            "A flat, leathery carapace without horny scutes, olive-green in young "
            "animals, with a pointed snout and a long neck."
        ),
        habitat_notes="Large rivers with sandy banks and calm pools; also tanks and wetlands.",
        distribution=(
            "Ganga, Brahmaputra, Mahanadi and other north Indian river systems, "
            "plus rivers of Pakistan and Nepal."
        ),
        diet="Omnivorous: fish, amphibians, carrion and aquatic plants.",
        behaviour="Spends long periods submerged or buried in sand; basks on banks and nests in sandy riverbanks.",
        ecological_role="Scavenger and predator that helps keep rivers clean; a cultural symbol in India.",
        threats="Poaching for meat and the pet trade, sand mining, river pollution and habitat loss.",
        conservation_status="Vulnerable",
        conservation_source=_IUCN,
        conservation_actions=(
            "Listed under Indian wildlife protection and CITES; protected river "
            "stretches, nest protection and awareness work support populations."
        ),
        facts=(
            "Softshell turtles have a flexible, leathery shell rather than hard horny scutes.",
        ),
        detection="broad",
        detection_note=(
            "The MegaFauna model has a broad 'turtle' class (mostly marine imagery); "
            "it cannot identify this freshwater species."
        ),
        references=_reptile_refs("Nilssonia gangetica"),
    ),
    _sp(
        id="indian_peacock_softshell",
        common_name="Indian peacock softshell turtle",
        scientific_name="Nilssonia hurum",
        group="Freshwater turtle",
        habitat="Freshwater",
        overview=(
            "The Indian peacock softshell is a medium-sized freshwater turtle named "
            "for the eye-like spots on its shell, which are most prominent in "
            "juveniles."
        ),
        identification="An olive softshell marked with dark eye-like (ocellus) patterns, especially in juveniles.",
        habitat_notes="Rivers, wetlands and ponds with soft beds.",
        distribution=(
            "Northern and central Indian rivers and wetlands, including the Ganga "
            "and Brahmaputra basins; also parts of Pakistan, Nepal and Bangladesh."
        ),
        diet="Omnivorous: fish, molluscs, aquatic plants and carrion.",
        behaviour="Buries in soft mud or sand and surfaces to bask; nests on riverbanks.",
        ecological_role="Aquatic omnivore in river and wetland food webs.",
        threats="Poaching for meat and trade, habitat degradation and sand mining.",
        conservation_status="Vulnerable",
        conservation_source=_IUCN,
        conservation_actions="Protected under Indian law and CITES; riverbank protection and anti-poaching work help.",
        facts=("The peacock-like eye-spots on juveniles fade as the animal grows.",),
        detection="broad",
        detection_note="Only the broad MegaFauna 'turtle' class applies; no species-level support.",
        references=_reptile_refs("Nilssonia hurum"),
    ),
    _sp(
        id="indian_narrow_headed_softshell",
        common_name="Indian narrow-headed softshell turtle",
        scientific_name="Chitra indica",
        group="Freshwater turtle",
        habitat="Freshwater",
        overview=(
            "The Indian narrow-headed softshell is one of Asia's largest freshwater "
            "turtles, with a very flat, streamlined shell built for life in large "
            "rivers."
        ),
        identification="A very flat softshell with a narrow head; adults can exceed a metre in shell length.",
        habitat_notes="Deep, large rivers with sandy beds; spends much time buried with only the eyes and snout exposed.",
        distribution="Large rivers of the Indian subcontinent, notably the Ganga and Brahmaputra systems.",
        diet="Carnivorous; ambushes fish from sandy river bottoms.",
        behaviour="A lie-in-wait predator that buries in sand; nests in riverbanks.",
        ecological_role="Large aquatic predator of major rivers; among the biggest freshwater turtles in Asia.",
        threats="Poaching, incidental capture in fishing gear, sand mining and river modification.",
        conservation_status="Endangered",
        conservation_source=_IUCN,
        conservation_actions="Protected under Indian law and CITES; nest protection and river conservation are used.",
        facts=("It buries itself in sand with only its eyes and snout exposed to ambush prey.",),
        detection="broad",
        detection_note="Only the broad MegaFauna 'turtle' class applies.",
        references=_reptile_refs("Chitra indica"),
    ),
    _sp(
        id="red_crowned_roofed_turtle",
        common_name="Red-crowned roofed turtle",
        scientific_name="Batagur kachuga",
        group="Freshwater turtle",
        habitat="Freshwater",
        overview=(
            "The red-crowned roofed turtle is a large, critically endangered river "
            "turtle of the Ganga system, famous for the brilliant head colours of "
            "breeding males."
        ),
        identification="A large hard-shelled river turtle; breeding males develop a striking red, yellow and blue head pattern.",
        habitat_notes="Deep, clear rivers with sandbanks for nesting, mainly the Chambal.",
        distribution="Ganga river system, mainly the Chambal and its tributaries in India and Nepal; highly localised.",
        diet="Herbivorous; grazes on aquatic vegetation.",
        behaviour="Grazes in the river and nests on sandbanks; males display their colours in the breeding season.",
        ecological_role="Riverine grazer; among the most threatened turtles in the world.",
        threats="Poaching, egg collection, fishing bycatch, sand mining and river modification.",
        conservation_status="Critically Endangered",
        conservation_source=_IUCN,
        conservation_actions="Protected under Indian law and CITES; dedicated nest-protection and head-starting programmes operate on the Chambal.",
        facts=("Its dramatic breeding colours make it one of the most colourful freshwater turtles.",),
        detection="broad",
        detection_note="Only the broad MegaFauna 'turtle' class applies.",
        references=_reptile_refs("Batagur kachuga"),
    ),
    _sp(
        id="indian_flapshell_turtle",
        common_name="Indian flapshell turtle",
        scientific_name="Lissemys punctata",
        group="Freshwater turtle",
        habitat="Freshwater",
        overview=(
            "The Indian flapshell turtle is the most frequently seen freshwater "
            "turtle of the Indian subcontinent, with soft skin-covered flaps over "
            "its hind limbs."
        ),
        identification="A small to medium olive-brown turtle whose plastron has soft, skin-covered flaps over the hind limbs.",
        habitat_notes="Ponds, marshes, tanks and slow rivers; tolerant of disturbed and seasonal wetlands.",
        distribution="Widespread across the Indian subcontinent, including India, Pakistan, Nepal, Bangladesh and Sri Lanka.",
        diet="Omnivorous: fish, frogs, insects, snails and plants.",
        behaviour="Often buries in mud and can remain dormant through dry spells; basks on banks.",
        ecological_role="A common wetland omnivore and scavenger that helps cycle nutrients.",
        threats="Collection for food and the pet trade, wetland loss and pollution.",
        conservation_status="Least Concern",
        conservation_source=_IUCN,
        conservation_actions="Protected under Indian law and CITES; common in many wetlands, but trade pressure continues.",
        facts=("The fleshy flaps over its hind limbs give it its name and help it bury in mud.",),
        detection="broad",
        detection_note="Only the broad MegaFauna 'turtle' class applies; no species-level support.",
        references=_reptile_refs("Lissemys punctata"),
    ),
    # ------------------------------------------------------------------ #
    # River dolphins and cetaceans
    # ------------------------------------------------------------------ #
    _sp(
        id="ganges_river_dolphin",
        common_name="Ganges river dolphin",
        scientific_name="Platanista gangetica",
        group="River dolphin & cetacean",
        habitat="Freshwater",
        overview=(
            "The Ganges river dolphin is a freshwater dolphin with a long slender "
            "snout and tiny, functionally blind eyes. It is the national aquatic "
            "animal of India and a flagship of great-river conservation."
        ),
        identification=(
            "A stocky grey-brown dolphin, about 2-2.5 m, with a long beak, a low "
            "triangular dorsal fin and very small eyes."
        ),
        habitat_notes="Deep pools and meanders of large, turbid rivers with strong currents; rarely enters the sea.",
        distribution=(
            "Ganga-Brahmaputra-Meghna and Karnaphuli river systems of India, Nepal "
            "and Bangladesh; a distinct subspecies survives in the Indus in Pakistan."
        ),
        diet="Carnivorous: fish, crustaceans and invertebrates caught in murky water using echolocation.",
        behaviour="Effectively blind; navigates and hunts by echolocation, often surfacing briefly to breathe; usually alone or in small groups.",
        ecological_role="Apex predator of large Indian rivers and a flagship indicator of river health.",
        threats="Bycatch in fishing gear, dams and barrages that fragment populations, pollution and water abstraction.",
        conservation_status="Endangered",
        conservation_source=_IUCN,
        conservation_actions=(
            "The national aquatic animal of India, protected under wildlife law, "
            "with dedicated sanctuaries and bycatch-reduction measures."
        ),
        facts=(
            "It is effectively blind and 'sees' by emitting ultrasonic clicks - a rare freshwater echolocator.",
            "India designated it the national aquatic animal in 2009.",
        ),
        detection="none",
        detection_note="No active model has a dolphin or river-dolphin class, so BlueEye cannot detect this species.",
        references=_mammal_refs("Platanista gangetica"),
    ),
    _sp(
        id="irrawaddy_dolphin",
        common_name="Irrawaddy dolphin",
        scientific_name="Orcaella brevirostris",
        group="River dolphin & cetacean",
        habitat="Coastal / marine",
        overview=(
            "The Irrawaddy dolphin is a round-headed coastal and estuarine dolphin, "
            "with a small freshwater population in the Mekong and a well-known "
            "population in Chilika Lagoon, India."
        ),
        identification="A slate-grey dolphin with a rounded head, a short beak and a flexible neck; up to about 2.7 m.",
        habitat_notes="Shallow coastal waters, estuaries and brackish lagoons; a few populations live in fresh water.",
        distribution=(
            "Coastal and estuarine waters of the Bay of Bengal and South-East Asia, "
            "including Chilika Lagoon in Odisha; freshwater populations occur in the "
            "Mekong, Mahakam and Ayeyarwady."
        ),
        diet="Fish, cephalopods and crustaceans.",
        behaviour="Often found in small groups near river mouths; known to spit water and, in some places, to associate with fishers.",
        ecological_role="Near-shore predator; the Chilika population is a focus of Indian conservation.",
        threats="Bycatch in gillnets, habitat degradation, dam construction and pollution.",
        conservation_status="Endangered",
        conservation_source=_IUCN,
        conservation_actions=(
            "Protected under Indian law; Chilika is a marine protected area and a "
            "dolphin reserve, with gillnet restrictions during key periods."
        ),
        facts=("It can spit water to herd fish and is known to cooperate with fishers in some regions.",),
        detection="none",
        detection_note="No active model has a dolphin class.",
        references=_mammal_refs("Orcaella brevirostris"),
    ),
    _sp(
        id="indian_ocean_humpback_dolphin",
        common_name="Indian Ocean humpback dolphin",
        scientific_name="Sousa plumbea",
        group="River dolphin & cetacean",
        habitat="Coastal / marine",
        overview=(
            "The Indian Ocean humpback dolphin is a coastal dolphin with a fleshy "
            "hump beneath its dorsal fin. It is one of the most threatened coastal "
            "dolphins in the world."
        ),
        identification="A grey dolphin with a long beak and a fleshy hump beneath the dorsal fin; about 2-2.5 m.",
        habitat_notes="Shallow coastal and estuarine waters, often close to shore.",
        distribution="Coastal waters of the western Indian Ocean, including the Arabian Sea coast of India.",
        diet="Fish, cephalopods and crustaceans taken in shallow coastal water.",
        behaviour="Occurs in small groups in near-shore waters and is often shy of boats.",
        ecological_role="Coastal predator that links inshore and estuarine food webs.",
        threats="Bycatch, coastal development, pollution and prey depletion.",
        conservation_status="Endangered",
        conservation_source=_IUCN,
        conservation_actions="Protected under Indian wildlife law; bycatch monitoring and coastal protected areas support conservation.",
        facts=("It is one of the most threatened coastal dolphins in the world.",),
        detection="none",
        detection_note="No active model has a dolphin class.",
        references=_mammal_refs("Sousa plumbea"),
    ),
    _sp(
        id="dugong",
        common_name="Dugong (sea cow)",
        scientific_name="Dugong dugon",
        group="River dolphin & cetacean",
        habitat="Coastal / marine",
        overview=(
            "The dugong is a large herbivorous marine mammal and India's only "
            "strictly herbivorous sea mammal. It grazes seagrass meadows in warm "
            "coastal waters."
        ),
        identification="A grey-brown marine mammal up to about 3 m, with paddle-like flippers and a fluked tail; no dorsal fin.",
        habitat_notes="Warm, shallow coastal waters with extensive seagrass beds.",
        distribution=(
            "Warm coastal waters of the Indo-Pacific, including the Gulf of Kutch, "
            "Gulf of Mannar and Andaman & Nicobar Islands in India."
        ),
        diet="Herbivorous; grazes almost entirely on seagrass.",
        behaviour="Slow-moving and largely solitary or in small groups; surfaces to breathe and can remain submerged for several minutes.",
        ecological_role="Seagrass grazer that shapes seagrass meadows and helps keep them productive.",
        threats="Loss and disturbance of seagrass habitat, entanglement in fishing gear, boat strikes and hunting.",
        conservation_status="Vulnerable",
        conservation_source=_IUCN,
        conservation_actions="Protected under Indian law; India has a Dugong Conservation Reserve (Palk Bay) and species recovery programmes.",
        facts=(
            "The dugong is the original 'mermaid' legend and India's only strictly herbivorous sea mammal.",
            "It is more closely related to elephants than to other marine mammals.",
        ),
        detection="none",
        detection_note="No active model has a dugong / sirenian class.",
        references=_mammal_refs("Dugong dugon"),
    ),
    # ------------------------------------------------------------------ #
    # Aquatic reptiles
    # ------------------------------------------------------------------ #
    _sp(
        id="gharial",
        common_name="Gharial",
        scientific_name="Gavialis gangeticus",
        group="Aquatic reptile",
        habitat="Freshwater",
        overview=(
            "The gharial is a critically endangered, fish-eating crocodilian with a "
            "long, narrow snout, now restricted to a few clear, free-flowing rivers "
            "of the Indian subcontinent."
        ),
        identification=(
            "A slender crocodilian with a long narrow snout lined with interlocking "
            "teeth; adult males bear a bulbous 'ghara' at the snout tip."
        ),
        habitat_notes="Deep, fast-flowing rivers with clean water and sandy banks for nesting.",
        distribution="A few rivers of the Indian subcontinent, chiefly the Chambal and Girwa in India and the Narayani in Nepal.",
        diet="Almost entirely piscivorous (fish-eating).",
        behaviour="Aquatic and largely fish-eating; males use the 'ghara' in courtship and females nest in sandbanks.",
        ecological_role="A specialised river predator and an indicator of healthy, free-flowing rivers.",
        threats="Dams and water diversion, fishing bycatch, sand mining, egg collection and habitat loss.",
        conservation_status="Critically Endangered",
        conservation_source=_IUCN,
        conservation_actions=(
            "Intensive Indian and Nepali conservation programmes combine nest "
            "protection, head-starting and river sanctuaries."
        ),
        facts=(
            "The male's 'ghara' is the only known sexual ornament in any crocodilian.",
            "Gharials are among the most thoroughly fish-eating of all crocodilians.",
        ),
        detection="none",
        detection_note=(
            "No installed model has a crocodilian or gharial class, so by default "
            "BlueEye cannot detect this species. The optional 'Open Images v7 "
            "Wildlife' model (downloadable on the Models page) has a generic "
            "'Crocodile' class that can flag crocodilians, but it cannot separate a "
            "gharial from a mugger or any other crocodilian - the species identity "
            "is never claimed."
        ),
        references=_reptile_refs("Gavialis gangeticus"),
    ),
    _sp(
        id="mugger_crocodile",
        common_name="Mugger (marsh crocodile)",
        scientific_name="Crocodylus palustris",
        group="Aquatic reptile",
        habitat="Freshwater",
        overview=(
            "The mugger, or marsh crocodile, is a broad-snouted crocodilian of "
            "Indian wetlands and one of the most widespread crocodilians in South Asia."
        ),
        identification="A broad-snouted grey-green crocodile, usually 2-4 m, distinguished by its relatively short, broad head.",
        habitat_notes="Freshwater marshes, lakes, reservoirs and rivers; tolerant of a range of water bodies.",
        distribution="Freshwater systems across the Indian subcontinent and Sri Lanka.",
        diet="Fish, turtles, birds and mammals; a generalist predator.",
        behaviour="Basks on banks and burrows into mud; unusually social for a crocodilian and can hunt cooperatively.",
        ecological_role="Top predator of Indian wetlands; helps regulate prey populations.",
        threats="Habitat loss, conflict with people, entanglement in fishing nets and poaching.",
        conservation_status="Vulnerable",
        conservation_source=_IUCN,
        conservation_actions=(
            "Protected under Indian law; captive breeding and reintroduction "
            "programmes (for example in central India) have helped recovery."
        ),
        facts=("It is the most social of crocodilians and can hunt in groups.",),
        detection="none",
        detection_note=(
            "No installed model has a crocodilian class, so by default BlueEye "
            "cannot detect this species. The optional 'Open Images v7 Wildlife' "
            "model (downloadable on the Models page) has a generic 'Crocodile' "
            "class that can flag crocodilians, but it does not establish that this "
            "is a mugger rather than a gharial or another crocodilian."
        ),
        references=_reptile_refs("Crocodylus palustris"),
    ),
    # ------------------------------------------------------------------ #
    # Freshwater sharks
    # ------------------------------------------------------------------ #
    _sp(
        id="ganges_shark",
        common_name="Ganges shark",
        scientific_name="Glyphis gangeticus",
        group="Freshwater shark",
        habitat="Freshwater",
        overview=(
            "The Ganges shark is a rare requiem shark known almost entirely from the "
            "Ganga-Brahmaputra system. It is one of the only sharks that lives "
            "permanently in fresh water and one of the world's most threatened sharks."
        ),
        identification="A stocky grey shark with a short, broadly rounded snout and small eyes; known from very few specimens.",
        habitat_notes="Large freshwater rivers and adjacent coastal waters.",
        distribution="Ganga-Brahmaputra river system and adjacent coastal waters; very rarely recorded.",
        diet="Fish and small crustaceans.",
        behaviour="Poorly known; presumed to be a riverine predator and easily confused with other requiem sharks.",
        ecological_role="A rare riverine top predator and one of the world's most threatened sharks.",
        threats="Fishing, bycatch, habitat alteration and pollution in one of the most densely populated river basins on Earth.",
        conservation_status="Critically Endangered",
        conservation_source=_IUCN,
        conservation_actions="Protected under Indian wildlife law and listed by CITES; research and river conservation are needed.",
        facts=(
            "It is one of the only sharks that lives permanently in fresh water and is known from remarkably few specimens.",
        ),
        detection="broad",
        detection_note=(
            "MegaFauna has a broad 'shark' class that cannot distinguish this "
            "species; no model is trained on Glyphis."
        ),
        references=_fish_refs("Glyphis gangeticus"),
    ),
    # ------------------------------------------------------------------ #
    # Indian coastal and marine fish
    # ------------------------------------------------------------------ #
    _sp(
        id="indian_oil_sardine",
        common_name="Indian oil sardine",
        scientific_name="Sardinella longiceps",
        group="Coastal & marine fish",
        habitat="Coastal / marine",
        overview=(
            "The Indian oil sardine is a small, abundant shoaling fish and the "
            "cornerstone of the Malabar-coast fishery. Its runs are closely tied to "
            "the monsoon."
        ),
        identification="A small silvery-green fish, usually 10-20 cm, with a single dark spot behind the gill cover.",
        habitat_notes="Coastal and shelf waters of the Arabian Sea, forming large surface shoals.",
        distribution="Arabian Sea and western Indian Ocean, especially the Malabar coast of India.",
        diet="Filter-feeder on plankton.",
        behaviour="Forms huge shoals; abundance and distribution fluctuate strongly with monsoon-driven upwelling.",
        ecological_role="Cornerstone of the Indian coastal fishery and a vital link between plankton and larger predators.",
        threats="Overfishing and climate-driven fluctuations in productivity.",
        conservation_status="Least Concern",
        conservation_source=_IUCN,
        conservation_actions="Managed through fishery regulations; the stock is monitored because of its economic importance.",
        facts=("Sardine shoals support huge traditional fisheries along the Kerala coast, with runs tied to the monsoon.",),
        detection="broad",
        detection_note="Only a generic 'fish' class exists; none of the models is trained on Indian marine fish.",
        references=_fish_refs("Sardinella longiceps"),
    ),
    _sp(
        id="indian_mackerel",
        common_name="Indian mackerel",
        scientific_name="Rastrelliger kanagurta",
        group="Coastal & marine fish",
        habitat="Coastal / marine",
        overview=(
            "The Indian mackerel is a streamlined, plankton-feeding fish and a major "
            "commercial catch along both Indian coasts."
        ),
        identification="A bluish-green body with dark wavy stripes, a forked tail and fine gill rakers; about 20-35 cm.",
        habitat_notes="Tropical coastal and shelf waters, often in schools near the surface.",
        distribution="Tropical Indo-Pacific, abundant along both the west and east coasts of India.",
        diet="Filter-feeder on plankton and small crustacean larvae.",
        behaviour="Shoals near the surface and feeds by swimming open-mouthed to sieve plankton.",
        ecological_role="Plankton-feeding forage fish that supports large fisheries and feeds predatory fish.",
        threats="Fishing pressure and environmental variability.",
        conservation_status="Least Concern",
        conservation_source=_IUCN,
        conservation_actions="Managed as a key commercial stock; catch and size regulations apply in India.",
        facts=("It feeds by swimming open-mouthed to sieve plankton from the water.",),
        detection="broad",
        detection_note="Only a generic 'fish' class exists.",
        references=_fish_refs("Rastrelliger kanagurta"),
    ),
    _sp(
        id="milkfish",
        common_name="Milkfish",
        scientific_name="Chanos chanos",
        group="Coastal & marine fish",
        habitat="Coastal / marine",
        overview=(
            "The milkfish is a hardy, silvery coastal fish and a mainstay of "
            "brackish-water aquaculture across the Indo-Pacific."
        ),
        identification="A silvery, lozenge-shaped body with a deeply forked tail and no teeth as an adult; can reach about 1.5 m.",
        habitat_notes="Coastal waters, lagoons and estuaries; farmed in brackish ponds.",
        distribution="Indo-Pacific; occurs in Indian coastal waters and is widely farmed.",
        diet="Omnivorous filter-feeder on algae and small invertebrates.",
        behaviour="Shoals in coastal waters and enters lagoons and estuaries; a fast, tireless swimmer.",
        ecological_role="Hardy coastal fish that converts algae into protein; a mainstay of brackish-water aquaculture.",
        threats="Not considered threatened; aquaculture is the dominant use.",
        conservation_status="Least Concern",
        conservation_source=_IUCN,
        conservation_actions="Not a conservation priority; sustained by extensive aquaculture.",
        facts=(
            "It is the only living species of its family, Chanidae, a lineage with fossil relatives over 100 million years old.",
        ),
        detection="broad",
        detection_note="Generic 'fish' class only; the brackish model is not trained on Indian species.",
        references=_fish_refs("Chanos chanos"),
    ),
    _sp(
        id="spotted_seahorse",
        common_name="Spotted seahorse (yellow seahorse)",
        scientific_name="Hippocampus kuda",
        group="Coastal & marine fish",
        habitat="Coastal / marine",
        overview=(
            "The spotted, or yellow, seahorse is a small armoured fish of seagrass "
            "beds and estuaries. It is heavily traded for traditional medicine and "
            "aquaria."
        ),
        identification="A small, heavily armoured fish with a horse-like head, a curled prehensile tail and mottled brown-yellow colouring.",
        habitat_notes="Seagrass beds, mangroves and estuaries, where it grips vegetation with its tail.",
        distribution=(
            "Indo-Pacific, including Indian coastal waters; population structure is "
            "complex and some populations are distinct."
        ),
        diet="Tiny crustaceans sucked up through a tubular snout.",
        behaviour="Secretive and slow-moving; males carry the developing young in a brood pouch.",
        ecological_role="Seagrass and mangrove dweller; traded for traditional medicine and aquaria, which drives overharvest.",
        threats="Overharvest for trade, bycatch and loss of seagrass and mangrove habitat.",
        conservation_status="Vulnerable",
        conservation_source=_IUCN,
        conservation_actions="Listed on CITES Appendix II; trade regulation and habitat protection are the main measures.",
        facts=(
            "Male seahorses become pregnant and give birth to the young.",
            "It anchors itself to seagrass with a prehensile tail rather than swimming continuously.",
        ),
        detection="none",
        detection_note="No active model has a seahorse class.",
        references=_fish_refs("Hippocampus kuda"),
    ),
    # ------------------------------------------------------------------ #
    # Sharks and rays
    # ------------------------------------------------------------------ #
    _sp(
        id="whale_shark",
        common_name="Whale shark",
        scientific_name="Rhincodon typus",
        group="Shark & ray",
        habitat="Coastal / marine",
        overview=(
            "The whale shark is the largest fish in the world, a harmless "
            "filter-feeder of warm surface waters. It is a major draw for "
            "ecotourism and a focus of Indian protection efforts."
        ),
        identification="Up to about 18 m, with a broad head and a white-spotted, checkerboard body pattern unique to each individual.",
        habitat_notes="Warm surface waters of tropical and temperate seas, often near plankton blooms.",
        distribution="Warm surface waters worldwide; regular along the Gujarat coast (Gulf of Kutch) and the Andaman Sea.",
        diet="Filter-feeder on plankton and small schooling fish.",
        behaviour="A slow, solitary filter-feeder that aggregates seasonally at feeding sites and migrates long distances.",
        ecological_role="Gentle ocean giant; a major draw for ecotourism and a focus of Indian protection efforts.",
        threats="Bycatch, boat strikes, unregulated tourism and targeted fishing in some regions.",
        conservation_status="Endangered",
        conservation_source=_IUCN,
        conservation_actions=(
            "Protected under Indian law and CITES; Gujarat is a global hotspot for "
            "whale-shark conservation and ecotourism regulation."
        ),
        facts=(
            "Each whale shark's spot pattern is unique, like a fingerprint, and is used by researchers to identify individuals.",
            "Despite its size it feeds only on plankton and small fish.",
        ),
        detection="broad",
        detection_note=(
            "MegaFauna and the aquarium model both have a broad 'shark' class, but "
            "neither distinguishes the whale shark from other sharks."
        ),
        references=_fish_refs("Rhincodon typus"),
    ),
    _sp(
        id="scalloped_hammerhead",
        common_name="Scalloped hammerhead",
        scientific_name="Sphyrna lewini",
        group="Shark & ray",
        habitat="Coastal / marine",
        overview=(
            "The scalloped hammerhead is a large, distinctive hammerhead shark of "
            "warm coastal and pelagic waters, heavily fished for its fins."
        ),
        identification="A grey shark with a curved 'hammer' head bearing eyes at the tips; typically 2-3 m.",
        habitat_notes="Warm coastal and pelagic waters; juveniles often use coastal nurseries.",
        distribution="Warm waters worldwide, including the Arabian Sea and Bay of Bengal.",
        diet="Fish, squid, octopus and rays.",
        behaviour="Forms schools, sometimes of hundreds, and makes long migrations; uses its head to detect hidden prey.",
        ecological_role="A coastal and pelagic predator, heavily fished for its fins, which are highly valued.",
        threats="Fishing and bycatch, driven largely by the fin trade.",
        conservation_status="Critically Endangered",
        conservation_source=_IUCN,
        conservation_actions="Listed on CITES Appendix II; protection, fin-trade regulation and bycatch reduction are the main needs.",
        facts=(
            "The wide head may improve the shark's electro-sensory detection of hidden prey.",
            "It sometimes gathers in large schools around seamounts.",
        ),
        detection="broad",
        detection_note="Only a broad 'shark' class exists; species identity is not established.",
        references=_fish_refs("Sphyrna lewini"),
    ),
    _sp(
        id="reef_manta_ray",
        common_name="Reef manta ray",
        scientific_name="Mobula alfredi",
        group="Shark & ray",
        habitat="Coastal / marine",
        overview=(
            "The reef manta ray is a large diamond-shaped filter-feeding ray of "
            "Indo-Pacific reefs and a focal species for marine tourism."
        ),
        identification="Up to about 5.5 m across, with horns beside the mouth and a distinctive white-and-dark dorsal pattern.",
        habitat_notes="Tropical reef and coastal waters, often around cleaning stations.",
        distribution="Tropical Indo-Pacific reefs, including the Andaman Sea and Lakshadweep.",
        diet="Filter-feeder on zooplankton.",
        behaviour="Often solitary or in loose groups; visits cleaning stations and performs barrel-roll feeding on plankton.",
        ecological_role="A keystone reef planktivore that moves nutrients between the water column and reefs, and a focal species for marine tourism.",
        threats="Bycatch, targeted fishing for gill plates, and disturbance by unregulated tourism.",
        conservation_status="Vulnerable",
        conservation_source=_IUCN,
        conservation_actions="Listed on CITES Appendix II; Indian protection and tourism guidance support conservation.",
        facts=(
            "Mantas are highly intelligent, appear to recognise themselves in mirrors and gather at cleaning stations.",
            "Each manta's belly-spot pattern is unique and is used to identify individuals.",
        ),
        detection="broad",
        detection_note=(
            "MegaFauna has a broad 'ray' class and the aquarium model a 'stingray' "
            "class, but neither identifies the species."
        ),
        references=_fish_refs("Mobula alfredi"),
    ),
    # ------------------------------------------------------------------ #
    # Sea turtles
    # ------------------------------------------------------------------ #
    _sp(
        id="olive_ridley_turtle",
        common_name="Olive Ridley sea turtle",
        scientific_name="Lepidochelys olivacea",
        group="Sea turtle",
        habitat="Coastal / marine",
        overview=(
            "The olive ridley is a small sea turtle famous for synchronous mass "
            "nesting, called arribadas. India hosts some of the world's largest "
            "nesting beaches."
        ),
        identification="A small, olive-grey sea turtle with a heart-shaped shell; 60-70 cm.",
        habitat_notes="Tropical and subtropical oceans; nests on sandy beaches, sometimes in huge synchronised groups.",
        distribution=(
            "Tropical oceans worldwide; India hosts major mass-nesting beaches at "
            "Gahirmatha and Rushikulya in Odisha."
        ),
        diet="Omnivorous: jellyfish, crabs, shrimp and fish.",
        behaviour="Famous for arribadas, in which thousands of females nest together over a few days; migrates long distances between feeding and nesting grounds.",
        ecological_role="A long-distance migrant that links ocean and beach ecosystems; a flagship of Indian turtle conservation.",
        threats="Bycatch in trawls and gillnets, beach development and light pollution, and egg predation.",
        conservation_status="Vulnerable",
        conservation_source=_IUCN,
        conservation_actions=(
            "Protected under Indian law; turtle-excluder-device efforts, nesting-beach "
            "protection and fishing bans near rookeries reduce bycatch."
        ),
        facts=("Odisha's beaches host arribadas - synchronised mass nestings where thousands of females come ashore together.",),
        detection="broad",
        detection_note="MegaFauna has a broad 'turtle' class that cannot separate the Olive Ridley from other sea turtles.",
        references=_reptile_refs("Lepidochelys olivacea"),
    ),
    _sp(
        id="green_sea_turtle",
        common_name="Green sea turtle",
        scientific_name="Chelonia mydas",
        group="Sea turtle",
        habitat="Coastal / marine",
        overview=(
            "The green sea turtle is a large sea turtle that grazes seagrass and "
            "algae as an adult. Its green body fat, not its shell, gives it its name."
        ),
        identification="A large sea turtle with a smooth greenish-brown carapace and a single pair of scales between the eyes.",
        habitat_notes="Tropical and subtropical seas, with seagrass meadows, reefs and nesting beaches.",
        distribution="Tropical and subtropical oceans, including the Lakshadweep and Andaman & Nicobar Islands of India.",
        diet="Herbivorous as an adult, grazing seagrass and algae; juveniles are omnivorous.",
        behaviour="Migrates between feeding grounds and nesting beaches; females return to nest on the beaches where they hatched.",
        ecological_role="A seagrass grazer that keeps meadows healthy and productive.",
        threats="Bycatch, poaching of eggs and adults, coastal development and fibropapillomatosis disease.",
        conservation_status="Endangered",
        conservation_source=_IUCN,
        conservation_actions="Protected under Indian law and CITES; nesting-beach protection and bycatch reduction support populations.",
        facts=("Its green body fat, not its shell colour, gives the species its name.",),
        detection="broad",
        detection_note="Only the broad MegaFauna 'turtle' class applies.",
        references=_reptile_refs("Chelonia mydas"),
    ),
    _sp(
        id="hawksbill_turtle",
        common_name="Hawksbill sea turtle",
        scientific_name="Eretmochelys imbricata",
        group="Sea turtle",
        habitat="Coastal / marine",
        overview=(
            "The hawksbill sea turtle is a critically endangered turtle with a "
            "beak-like mouth and a coveted 'tortoiseshell' shell. It feeds mainly "
            "on sponges on coral reefs."
        ),
        identification="A sea turtle with a beak-like mouth and overlapping, mottled amber-and-brown shell plates.",
        habitat_notes="Coral reefs, rocky areas and lagoons in tropical seas.",
        distribution="Tropical reefs worldwide; nests on Indian islands and some mainland beaches.",
        diet="Spongivore; feeds mainly on sea sponges on coral reefs.",
        behaviour="Forages on reefs and nests on sandy beaches, often in small numbers; highly migratory.",
        ecological_role="Helps maintain reef health by controlling sponge populations.",
        threats="Historic and continuing exploitation for tortoiseshell, bycatch, egg poaching and reef degradation.",
        conservation_status="Critically Endangered",
        conservation_source=_IUCN,
        conservation_actions="Protected under Indian law and CITES; anti-trade enforcement and nesting-beach protection are the main measures.",
        facts=("Its 'tortoiseshell' shell made it one of the most exploited turtles in history.",),
        detection="broad",
        detection_note="Only the broad MegaFauna 'turtle' class applies.",
        references=_reptile_refs("Eretmochelys imbricata"),
    ),
    _sp(
        id="leatherback_turtle",
        common_name="Leatherback sea turtle",
        scientific_name="Dermochelys coriacea",
        group="Sea turtle",
        habitat="Coastal / marine",
        overview=(
            "The leatherback is the largest sea turtle and the only one without a "
            "hard shell. It is a jellyfish specialist capable of deep, long-distance "
            "migrations."
        ),
        identification="Up to about 2 m, with a leathery, ridged shell instead of hard scutes.",
        habitat_notes="Open oceans and temperate feeding grounds; nests on tropical sandy beaches.",
        distribution="All oceans; nests on some Indian beaches, for example in the Andaman & Nicobar Islands.",
        diet="Jellyfish specialist; also eats other soft-bodied prey.",
        behaviour="Makes the longest migrations of any reptile and can dive deeper than 1,000 m; regulates its body temperature in cold water.",
        ecological_role="Controls jellyfish populations across ocean basins.",
        threats="Bycatch, egg poaching, coastal development and ingestion of marine plastics.",
        conservation_status="Vulnerable",
        conservation_source=_IUCN,
        conservation_actions="Protected under Indian law and CITES; bycatch reduction and nesting-beach protection are the main measures.",
        facts=(
            "It can dive deeper than 1,000 m and regulate its body temperature in cold water.",
            "It feeds almost entirely on jellyfish, which are nutritionally poor.",
        ),
        detection="broad",
        detection_note="Only the broad MegaFauna 'turtle' class applies; species identity is not established.",
        references=_reptile_refs("Dermochelys coriacea"),
    ),
    # ------------------------------------------------------------------ #
    # Reef fish with explicit class support
    # ------------------------------------------------------------------ #
    _sp(
        id="humphead_wrasse",
        common_name="Humphead wrasse",
        scientific_name="Cheilinus undulatus",
        group="Reef fish",
        habitat="Reef",
        overview=(
            "The humphead wrasse is one of the largest reef fishes, a slow-growing "
            "apex predator of Indo-Pacific reefs. It is naturally rare and "
            "vulnerable to overfishing."
        ),
        identification=(
            "A very large, thick-lipped reef fish reaching over 2 m, with a bulbous "
            "forehead that grows more pronounced with age; males are electric "
            "blue-green, females mottled."
        ),
        habitat_notes="Coral reef slopes and lagoon reefs, usually in low densities.",
        distribution="Indo-Pacific coral reefs, including the Lakshadweep and Andaman reefs of India.",
        diet="Carnivorous; crushes hard-shelled prey such as molluscs, sea stars and crustaceans.",
        behaviour="A long-lived protogynous hermaphrodite (changes sex during life) that moves among reefs and forms spawning aggregations.",
        ecological_role="Apex reef predator; one of the largest reef fishes, and naturally rare.",
        threats="Fishing for the live-reef-fish food trade, slow growth and spawning-aggregation fishing.",
        conservation_status="Endangered",
        conservation_source=_IUCN,
        conservation_actions="Listed on CITES Appendix II and protected in some countries; export and catch regulations are important.",
        facts=("It changes sex during life - most large, colourful individuals are males that were born female.",),
        detection="direct",
        detection_note="The Fish & Invertebrates model includes an explicit 'cheilinus_undulatus' class for this species.",
        references=_fish_refs("Cheilinus undulatus"),
    ),
    _sp(
        id="humphead_parrotfish",
        common_name="Humphead parrotfish",
        scientific_name="Bolbometopon muricatum",
        group="Reef fish",
        habitat="Reef",
        overview=(
            "The humphead parrotfish is the largest parrotfish, a coral-scraping "
            "herbivore that shapes reef structure and produces coral sand."
        ),
        identification="Up to about 1.3 m, with a distinctive bulbous forehead and fused beak-like teeth.",
        habitat_notes="Indo-Pacific reef slopes and lagoons, often in schools.",
        distribution="Indo-Pacific reef slopes, including Indian reef areas.",
        diet="Herbivorous; scrapes live coral and algae, a major bioeroder of reefs.",
        behaviour="Schoals by day and sleeps in reef crevices at night, some individuals producing a mucus sleeping cocoon.",
        ecological_role="Produces coral sand and shapes reef structure; ecologically vital and heavily overfished.",
        threats="Night-time spearfishing at sleeping sites, overfishing and reef degradation.",
        conservation_status="Vulnerable",
        conservation_source=_IUCN,
        conservation_actions="Protected in some countries and managed through reef-fishery regulations; spawning aggregations need protection.",
        facts=("A single large individual can produce hundreds of kilograms of coral sand a year.",),
        detection="direct",
        detection_note="The Fish & Invertebrates model includes an explicit 'bolbometopon_muricatum' class.",
        references=_fish_refs("Bolbometopon muricatum"),
    ),
    _sp(
        id="camouflage_grouper",
        common_name="Camouflage grouper (humpback grouper)",
        scientific_name="Cromileptes altivelis",
        group="Reef fish",
        habitat="Reef",
        overview=(
            "The camouflage grouper, or humpback grouper, is a high-arched reef "
            "grouper with a pattern of round black spots. Its popularity in the live "
            "food-fish trade pressures wild stocks."
        ),
        identification="A pale grey-white grouper with a high-arched back and a scattering of round black spots.",
        habitat_notes="Coral reefs and lagoons, sheltering in caves and under ledges.",
        distribution="Indo-Pacific reefs, including the Andaman and Nicobar Islands.",
        diet="Carnivorous; ambushes small fish and crustaceans among coral.",
        behaviour="An ambush predator that shelters by day and hunts around the reef; protogynous (changes sex during life).",
        ecological_role="Reef predator; popular in the live food-fish trade, which pressures wild stocks.",
        threats="Fishing for the live food-fish trade and reef habitat degradation.",
        conservation_status="Vulnerable",
        conservation_source=_IUCN,
        conservation_actions="Aquaculture for the trade is being developed; reef protection and trade regulation support wild stocks.",
        facts=("Its spot pattern gives it the alternative name 'panther grouper'.",),
        detection="direct",
        detection_note="The Fish & Invertebrates model includes an explicit 'cromileptes_altivelis' class.",
        references=_fish_refs("Cromileptes altivelis"),
    ),
    # ------------------------------------------------------------------ #
    # Invertebrates
    # ------------------------------------------------------------------ #
    _sp(
        id="sandfish_sea_cucumber",
        common_name="Sandfish sea cucumber",
        scientific_name="Holothuria scabra",
        group="Invertebrate",
        habitat="Coastal / marine",
        overview=(
            "The sandfish is a leathery deposit-feeding sea cucumber of Indo-Pacific "
            "coastal flats. It is heavily harvested for the beche-de-mer trade and is "
            "important for sediment health."
        ),
        identification="An elongated, leathery sea cucumber, usually 15-30 cm, grey to olive with darker wrinkles.",
        habitat_notes="Shallow coastal and lagoonal sediments where it buries during the day.",
        distribution="Indo-Pacific shallow coastal flats, including Indian shores and the Gulf of Mannar.",
        diet="Deposit-feeder; swallows sediment and extracts organic matter.",
        behaviour="Burrows into sediment and emerges to feed; can expel and regenerate its internal organs when threatened.",
        ecological_role="Recycles nutrients and aerates seabed sediments; heavily harvested for the beche-de-mer trade.",
        threats="Overharvest for the dried sea-cucumber trade and habitat degradation.",
        conservation_status="Endangered",
        conservation_source=_IUCN,
        conservation_actions="Listed on CITES Appendix II; India regulates sea-cucumber collection and is developing aquaculture.",
        facts=("When threatened it can expel its internal organs and later regrow them.",),
        detection="broad",
        detection_note="The Fish & Invertebrates model has a generic 'sea_cucumber' class but does not identify the species.",
        references=_invert_refs("Holothuria scabra"),
    ),
    _sp(
        id="mud_crab",
        common_name="Giant mud crab",
        scientific_name="Scylla serrata",
        group="Invertebrate",
        habitat="Brackish / estuarine",
        overview=(
            "The giant mud crab is a large, powerful crab of Indo-Pacific mangroves "
            "and estuaries, and an important aquaculture species."
        ),
        identification="A large, heavy crab with a smooth greenish-brown to nearly black shell and powerful claws.",
        habitat_notes="Mangroves, estuaries, mudflats and backwaters.",
        distribution="Indo-Pacific mangroves and estuaries, including India's Sundarbans and coastal backwaters.",
        diet="Omnivorous scavenger and predator of molluscs and small crabs.",
        behaviour="Burrows in mud and is active mainly at night; males grow larger than females.",
        ecological_role="An important mangrove and estuarine predator and scavenger, and a valuable aquaculture species.",
        threats="Not formally assessed; local pressures include overharvest and mangrove loss.",
        conservation_status="Not Evaluated",
        conservation_source=_NE,
        conservation_actions="Not assessed; mangrove protection and aquaculture-based production support sustainability.",
        facts=("Males grow much larger than females and are the main target of mud-crab aquaculture.",),
        detection="broad",
        detection_note="The brackish model has a generic 'crab' class but cannot identify the species.",
        references=_invert_refs("Scylla serrata"),
    ),
    _sp(
        id="crown_of_thorns",
        common_name="Crown-of-thorns starfish",
        scientific_name="Acanthaster planci",
        group="Invertebrate",
        habitat="Reef",
        overview=(
            "The crown-of-thorns starfish is a large, venomous coral-eating starfish "
            "of Indo-Pacific reefs. Population outbreaks can cause widespread reef "
            "decline."
        ),
        identification="A large multi-armed starfish up to about 35 cm, covered in venomous spines.",
        habitat_notes="Coral reefs across the Indo-Pacific.",
        distribution="Indo-Pacific coral reefs, including Indian reefs.",
        diet="Coral; feeds by everting its stomach over coral polyps.",
        behaviour="A specialised coral predator; outbreaks are often linked to nutrient enrichment and predator removal.",
        ecological_role="A natural coral predator whose outbreaks can devastate reefs.",
        threats="Not formally assessed as a species; outbreaks threaten reef ecosystems rather than the starfish itself.",
        conservation_status="Not Evaluated",
        conservation_source=_NE,
        conservation_actions="Not assessed; reef managers control outbreaks manually in some protected areas.",
        facts=(
            "During outbreaks it can strip a reef of living coral, driving major reef declines.",
            "It is one of the few starfish that feeds on live hard coral.",
        ),
        detection="direct",
        detection_note="The Fish & Invertebrates model includes an explicit 'crown_of_thorns' class.",
        references=_invert_refs("Acanthaster planci"),
    ),
)


# --------------------------------------------------------------------------- #
# Verified species photographs (generated by scripts/fetch_species_images.py)
# --------------------------------------------------------------------------- #
try:  # photographs are an enhancement, never a hard dependency
    from app.content.species_images import SPECIES_IMAGES as _SPECIES_IMAGES
except ImportError:  # pragma: no cover - only while the generated file is absent
    _SPECIES_IMAGES = {}


def _with_photo(species: Species) -> Species:
    """Attach a verified photograph + attribution when one is available.

    A species with no verified image is returned unchanged, so the UI keeps
    its icon fallback and never shows a broken or invented picture.
    """
    info = _SPECIES_IMAGES.get(species.id) or {}
    url = str(info.get("url") or "").strip()
    if not url:
        return species
    credit = str(info.get("credit") or "Wikimedia Commons").strip()
    licence = str(info.get("license") or "").strip()
    attribution = f"{credit} · {licence}" if licence else credit
    return replace(species, image=url, image_credit=attribution)


SPECIES: tuple[Species, ...] = tuple(_with_photo(item) for item in SPECIES)


# --------------------------------------------------------------------------- #
# Lookup / filtering helpers (used by the UI and covered by tests)
# --------------------------------------------------------------------------- #
def search_species(
    query: str = "",
    species: Sequence[Species] = SPECIES,
) -> list[Species]:
    """Case-insensitive match on common or scientific name (and group)."""
    text = (query or "").strip().lower()
    if not text:
        return list(species)
    return [
        item
        for item in species
        if text in item.common_name.lower()
        or text in item.scientific_name.lower()
        or text in item.group.lower()
    ]


def filter_species(
    query: str = "",
    habitats: Iterable[str] = (),
    groups: Iterable[str] = (),
    species: Sequence[Species] = SPECIES,
) -> list[Species]:
    """Apply search + habitat + group filters (each filter is OR within itself)."""
    habitat_set = {value for value in habitats}
    group_set = {value for value in groups}
    return [
        item
        for item in search_species(query, species)
        if (not habitat_set or item.habitat in habitat_set)
        and (not group_set or item.group in group_set)
    ]


def get_species(species_id: str) -> Species | None:
    """Look a species up by its stable id."""
    for item in SPECIES:
        if item.id == species_id:
            return item
    return None


def all_habitats(species: Sequence[Species] = SPECIES) -> list[str]:
    return sorted({item.habitat for item in species})


def all_groups(species: Sequence[Species] = SPECIES) -> list[str]:
    return sorted({item.group for item in species})


__all__ = [
    "DETECTION_CATEGORIES",
    "DETECTION_ORDER",
    "Species",
    "SPECIES",
    "search_species",
    "filter_species",
    "get_species",
    "all_habitats",
    "all_groups",
]
