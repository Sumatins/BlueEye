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

Biological facts (names, distribution, diet, IUCN status) are drawn from
public conservation/research sources; the conservation status field names the
source and the assessment context. Facts that could not be confirmed are
marked ``(requires verification)`` rather than guessed.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class Species:
    """One aquatic species profile (educational reference data)."""

    id: str
    common_name: str
    scientific_name: str
    group: str
    habitat: str
    distribution: str
    appearance: str
    diet: str
    ecological_role: str
    conservation_status: str
    conservation_source: str
    interesting_fact: str
    detection: str
    detection_note: str
    image: str = ""
    image_credit: str = ""

    @property
    def detection_label(self) -> str:
        return DETECTION_CATEGORIES.get(self.detection, "Unverified")


def _sp(**kwargs) -> Species:
    return Species(**kwargs)


_IUCN = "IUCN Red List of Threatened Species (assessments as published up to 2024)"
_NE = "Not Evaluated (IUCN Red List of Threatened Species)"

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
        distribution="Rivers and floodplain waters of the Indian subcontinent (Ganga-Brahmaputra and peninsular systems); widely farmed across South Asia.",
        appearance="Large silvery carp reaching about 1–2 m; reddish fins, a blunt head and a single pair of barbels.",
        diet="Mostly herbivorous: plankton, algae, submerged vegetation and organic detritus.",
        ecological_role="Mid-water grazer that converts plant matter into fish biomass; a key food fish and pillar of South Asian aquaculture.",
        conservation_status="Least Concern",
        conservation_source=_IUCN,
        interesting_fact="Along with catla and mrigal it forms the 'Indian major carp' trio that underpins freshwater aquaculture in the region.",
        detection="broad",
        detection_note="Only a generic 'fish' class exists (in the brackish / underwater / aquarium models). None is trained on Indian freshwater species, so the model cannot confirm this is Rohu.",
    ),
    _sp(
        id="catla",
        common_name="Catla",
        scientific_name="Labeo catla",
        group="Freshwater fish",
        habitat="Freshwater",
        distribution="Native to the Ganga, Brahmaputra and Mahanadi river systems; introduced widely for aquaculture.",
        appearance="Deep-bodied, large-headed silver carp with a prominent lower jaw; grows to about 1.8 m and is the fastest-growing Indian major carp.",
        diet="Zooplankton and phytoplankton filter-feeder, feeding near the surface.",
        ecological_role="Surface filter-feeder; a staple aquaculture and capture-fishery species.",
        conservation_status="Least Concern",
        conservation_source=_IUCN,
        interesting_fact="It was previously placed in its own genus, Catla, and is now classified as Labeo catla.",
        detection="broad",
        detection_note="Falls under a generic 'fish' class only; no active model is specific to Indian carps.",
    ),
    _sp(
        id="mrigal",
        common_name="Mrigal",
        scientific_name="Cirrhinus mrigala",
        group="Freshwater fish",
        habitat="Freshwater",
        distribution="Indian subcontinent rivers, especially the Ganga and peninsular systems; farmed throughout the region.",
        appearance="Slender, greyish-silver carp with a pointed head and a single short barbel; typically 30–100 cm.",
        diet="Bottom-dwelling detritivore feeding on decaying plant matter, algae and mud-dwelling organisms.",
        ecological_role="Benthic detritivore that recycles organic matter; the third 'Indian major carp'.",
        conservation_status="Least Concern",
        conservation_source=_IUCN,
        interesting_fact="Its habit of grubbing the pond bottom keeps organic waste in check in polyculture ponds.",
        detection="broad",
        detection_note="Covered only by a generic 'fish' class; no Indian-specific model.",
    ),
    _sp(
        id="hilsa",
        common_name="Hilsa (Indian shad)",
        scientific_name="Tenualosa ilisha",
        group="Freshwater fish",
        habitat="Freshwater",
        distribution="Ganga-Brahmaputra deltas and coastal waters of the northern Indian Ocean; famous from the Hooghly and Padma.",
        appearance="Laterally compressed, silver-bodied herring-like fish with a saw-toothed belly keel; usually 30–50 cm.",
        diet="Plankton-feeder as an adult, feeding in estuarine and coastal waters.",
        ecological_role="Anadromous fish migrating up rivers to spawn; a culturally and economically vital fishery.",
        conservation_status="Least Concern",
        conservation_source=_IUCN,
        interesting_fact="Hilsa runs are celebrated in Bengal and Bangladesh; the fish is the national fish of Bangladesh.",
        detection="broad",
        detection_note="Only a generic 'fish' class is available; species identity is not established.",
    ),
    _sp(
        id="golden_mahseer",
        common_name="Golden mahseer",
        scientific_name="Tor putitora",
        group="Freshwater fish",
        habitat="Freshwater",
        distribution="Foothill rivers of the Himalaya and northern India (Ganga, Brahmaputra, Indus drainages).",
        appearance="Powerful, golden-olive cyprinid with large scales and broad fins; can exceed 2 m and 50 kg.",
        diet="Omnivorous: small fish, crustaceans, insects and fruit fallen into rivers.",
        ecological_role="Top freshwater predator and a flagship of Himalayan river conservation; a prized sport and food fish.",
        conservation_status="Endangered",
        conservation_source=_IUCN,
        interesting_fact="One of the largest freshwater carps in the world and a beloved 'tiger of the river' for anglers.",
        detection="broad",
        detection_note="Only a generic 'fish' class exists; no model is trained on mahseer.",
    ),
    _sp(
        id="chocolate_mahseer",
        common_name="Chocolate mahseer",
        scientific_name="Neolissochilus hexagonolepis",
        group="Freshwater fish",
        habitat="Freshwater",
        distribution="Hill streams of the eastern Himalaya and north-east India (Brahmaputra tributaries).",
        appearance="Deep-bodied olive-brown fish with large scales; reaches about 60 cm.",
        diet="Omnivorous drift-feeder on insects, algae and small invertebrates.",
        ecological_role="Stream fish important to local food security and angling in the eastern Himalaya.",
        conservation_status="Near Threatened",
        conservation_source=_IUCN,
        interesting_fact="In Meghalaya and Arunachal Pradesh it is a traditional food fish caught by indigenous river communities.",
        detection="broad",
        detection_note="Only a generic 'fish' class exists.",
    ),
    _sp(
        id="deccan_mahseer",
        common_name="Deccan mahseer (blue-finned mahseer)",
        scientific_name="Tor khudree",
        group="Freshwater fish",
        habitat="Freshwater",
        distribution="Fast-flowing rivers of peninsular India, especially the Western Ghats.",
        appearance="Stocky, bluish-grey mahseer reaching about 1.5 m.",
        diet="Omnivorous; feeds on fish, insects, molluscs and fruits.",
        ecological_role="Apex river predator in Ghat streams; important for ecotourism and sport fishing.",
        conservation_status="Endangered",
        conservation_source=_IUCN,
        interesting_fact="Released by anglers after capture, it supports a catch-and-release sport fishery in the Cauvery.",
        detection="broad",
        detection_note="Only a generic 'fish' class exists; no species-specific model.",
    ),
    _sp(
        id="striped_snakehead",
        common_name="Striped snakehead (murrel)",
        scientific_name="Channa striata",
        group="Freshwater fish",
        habitat="Freshwater",
        distribution="Widespread across South and South-East Asia, including most Indian freshwater systems.",
        appearance="Elongated, snake-like grey-brown body with darker bands; can breathe air using a labyrinth organ.",
        diet="Carnivorous ambush predator of fish, frogs, insects and crustaceans.",
        ecological_role="Predator that structures prey communities in ponds and wetlands; widely cultured and valued as food.",
        conservation_status="Least Concern",
        conservation_source=_IUCN,
        interesting_fact="It can survive in oxygen-poor water by gulping air at the surface.",
        detection="broad",
        detection_note="Only a generic 'fish' class exists.",
    ),
    _sp(
        id="indian_featherback",
        common_name="Indian featherback (clown knifefish)",
        scientific_name="Chitala chitala",
        group="Freshwater fish",
        habitat="Freshwater",
        distribution="Rivers and floodplains of the Indian subcontinent, including the Ganga and Brahmaputra basins.",
        appearance="Deep, knife-shaped silvery body with a row of dark spots along the flank and a long anal fin.",
        diet="Carnivorous; feeds on fish and invertebrates, hunting at dusk and night.",
        ecological_role="Predatory fish of slow rivers and floodplains; important in local fisheries.",
        conservation_status="Near Threatened",
        conservation_source=_IUCN,
        interesting_fact="Despite its 'knifefish' name it is not related to the South American knifefishes.",
        detection="broad",
        detection_note="Only a generic 'fish' class exists.",
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
        distribution="Ganga, Brahmaputra, Mahanadi and other north Indian river systems.",
        appearance="Flat leathery carapace without scutes, olive-green in young animals; pointed snout and long neck.",
        diet="Omnivorous: fish, amphibians, carrion and aquatic plants.",
        ecological_role="Scavenger and predator that helps keep rivers clean; a cultural symbol in India.",
        conservation_status="Vulnerable",
        conservation_source=_IUCN,
        interesting_fact="Softshell turtles have a flexible, leathery shell rather than hard horny scutes.",
        detection="broad",
        detection_note="The MegaFauna model has a broad 'turtle' class (mostly marine imagery); it cannot identify this freshwater species.",
    ),
    _sp(
        id="indian_peacock_softshell",
        common_name="Indian peacock softshell turtle",
        scientific_name="Nilssonia hurum",
        group="Freshwater turtle",
        habitat="Freshwater",
        distribution="Northern and central Indian rivers and wetlands.",
        appearance="Olive softshell marked with dark eye-like (ocellus) patterns, especially in juveniles.",
        diet="Omnivorous: fish, molluscs, aquatic plants and carrion.",
        ecological_role="Aquatic omnivore in river and wetland food webs.",
        conservation_status="Vulnerable",
        conservation_source=_IUCN,
        interesting_fact="The peacock-like eye-spots on juveniles fade as the animal grows.",
        detection="broad",
        detection_note="Only the broad MegaFauna 'turtle' class applies; no species-level support.",
    ),
    _sp(
        id="indian_narrow_headed_softshell",
        common_name="Indian narrow-headed softshell turtle",
        scientific_name="Chitra indica",
        group="Freshwater turtle",
        habitat="Freshwater",
        distribution="Large rivers of the Indian subcontinent, notably the Ganga and Brahmaputra.",
        appearance="Very flat, streamlined softshell with a narrow head; adults can reach over a metre in shell length.",
        diet="Carnivorous; ambushes fish from sandy river bottoms.",
        ecological_role="Large aquatic predator; one of Asia's biggest freshwater turtles.",
        conservation_status="Endangered",
        conservation_source=_IUCN,
        interesting_fact="It buries itself in sand with only its eyes and snout exposed to ambush prey.",
        detection="broad",
        detection_note="Only the broad MegaFauna 'turtle' class applies.",
    ),
    _sp(
        id="red_crowned_roofed_turtle",
        common_name="Red-crowned roofed turtle",
        scientific_name="Batagur kachuga",
        group="Freshwater turtle",
        habitat="Freshwater",
        distribution="Ganga river system, mainly the Chambal and its tributaries.",
        appearance="Large hard-shelled river turtle; breeding males develop a striking red, yellow and blue head pattern.",
        diet="Herbivorous; grazes on aquatic vegetation.",
        ecological_role="Riverine grazer; among the most threatened turtles in the world.",
        conservation_status="Critically Endangered",
        conservation_source=_IUCN,
        interesting_fact="Its dramatic breeding colours make it one of the most colourful freshwater turtles.",
        detection="broad",
        detection_note="Only the broad MegaFauna 'turtle' class applies.",
    ),
    _sp(
        id="indian_flapshell_turtle",
        common_name="Indian flapshell turtle",
        scientific_name="Lissemys punctata",
        group="Freshwater turtle",
        habitat="Freshwater",
        distribution="Widespread across the Indian subcontinent in ponds, marshes and slow rivers.",
        appearance="Small to medium olive-brown turtle whose plastron (underside) has soft, skin-covered flaps.",
        diet="Omnivorous: fish, frogs, insects, snails and plants.",
        ecological_role="Common wetland omnivore and scavenger; the most frequently seen Indian freshwater turtle.",
        conservation_status="Least Concern",
        conservation_source=_IUCN,
        interesting_fact="The fleshy flaps over its hind limbs give it its name and help it bury in mud.",
        detection="broad",
        detection_note="Only the broad MegaFauna 'turtle' class applies; no species-level support.",
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
        distribution="Ganga-Brahmaputra-Meghna and Karnaphuli river systems of India, Nepal and Bangladesh.",
        appearance="Stocky, grey-brown freshwater dolphin with a long slender snout and tiny, functionally blind eyes.",
        diet="Carnivorous: fish, crustaceans and invertebrates caught in murky water using echolocation.",
        ecological_role="Apex predator of large Indian rivers and a flagship indicator of river health.",
        conservation_status="Endangered",
        conservation_source=_IUCN,
        interesting_fact="It is effectively blind and 'sees' by emitting ultrasonic clicks — a rare freshwater echolocator.",
        detection="none",
        detection_note="No active model has a dolphin or river-dolphin class, so BlueEye cannot detect this species.",
    ),
    _sp(
        id="irrawaddy_dolphin",
        common_name="Irrawaddy dolphin",
        scientific_name="Orcaella brevirostris",
        group="River dolphin & cetacean",
        habitat="Coastal / marine",
        distribution="Coastal and estuarine waters of the Bay of Bengal and South-East Asia; a small freshwater population lives in the Mekong and Chilika lagoon hosts a population.",
        appearance="Round-headed, slate-grey dolphin with a short beak and a flexible neck.",
        diet="Fish, cephalopods and crustaceans.",
        ecological_role="Near-shore predator; some populations live in estuaries and brackish lagoons such as Chilika in Odisha.",
        conservation_status="Endangered",
        conservation_source=_IUCN,
        interesting_fact="It can spit water to herd fish and is known to cooperate with fishers in some regions.",
        detection="none",
        detection_note="No active model has a dolphin class.",
    ),
    _sp(
        id="indian_ocean_humpback_dolphin",
        common_name="Indian Ocean humpback dolphin",
        scientific_name="Sousa plumbea",
        group="River dolphin & cetacean",
        habitat="Coastal / marine",
        distribution="Coastal waters of the Indian Ocean, including the Arabian Sea coast of India.",
        appearance="Grey dolphin with a characteristic fleshy hump beneath the dorsal fin and a long beak.",
        diet="Fish, cephalopods and crustaceans taken in shallow coastal water.",
        ecological_role="Coastal predator, often found close to shore and estuaries.",
        conservation_status="Endangered",
        conservation_source=_IUCN,
        interesting_fact="It is one of the most threatened coastal dolphins in the world.",
        detection="none",
        detection_note="No active model has a dolphin class.",
    ),
    _sp(
        id="dugong",
        common_name="Dugong (sea cow)",
        scientific_name="Dugong dugon",
        group="River dolphin & cetacean",
        habitat="Coastal / marine",
        distribution="Warm coastal waters of the Indo-Pacific, including the Gulf of Kutch, Gulf of Mannar and Andaman & Nicobar Islands.",
        appearance="Large grey-brown marine mammal with a fluked tail and paddle-like flippers; can reach 3 m.",
        diet="Herbivorous; grazes almost entirely on seagrass.",
        ecological_role="Seagrass grazer that shapes seagrass meadows; India's only herbivorous marine mammal.",
        conservation_status="Vulnerable",
        conservation_source=_IUCN,
        interesting_fact="The dugong is the original 'mermaid' legend and India's only strictly herbivorous sea mammal.",
        detection="none",
        detection_note="No active model has a dugong / sirenian class.",
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
        distribution="A few clear, fast-flowing rivers of the Indian subcontinent, chiefly the Chambal and Girwa in India and the Narayani in Nepal.",
        appearance="Slender crocodilian with a long, narrow snout lined with interlocking teeth; adult males bear a bulbous 'ghara' on the snout tip.",
        diet="Almost entirely piscivorous (fish-eating).",
        ecological_role="Specialised river predator; an indicator of healthy, free-flowing rivers.",
        conservation_status="Critically Endangered",
        conservation_source=_IUCN,
        interesting_fact="The male's 'ghara' is the only known sexual ornament in any crocodilian.",
        detection="none",
        detection_note="No active model has a crocodilian or gharial class.",
    ),
    _sp(
        id="mugger_crocodile",
        common_name="Mugger (marsh crocodile)",
        scientific_name="Crocodylus palustris",
        group="Aquatic reptile",
        habitat="Freshwater",
        distribution="Freshwater marshes, lakes and rivers across the Indian subcontinent and Sri Lanka.",
        appearance="Broad-snouted grey-green crocodile, usually 2–4 m.",
        diet="Fish, turtles, birds and mammals; a generalist predator.",
        ecological_role="Top predator of Indian wetlands and one of the most widespread crocodilians.",
        conservation_status="Vulnerable",
        conservation_source=_IUCN,
        interesting_fact="It is the most social of crocodilians and can hunt in groups.",
        detection="none",
        detection_note="No active model has a crocodilian class.",
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
        distribution="Ganga-Brahmaputra river system and adjacent coastal waters; very rarely recorded.",
        appearance="Stocky grey requiem shark with a short, broadly rounded snout and small eyes.",
        diet="Fish and small crustaceans.",
        ecological_role="Rare riverine predator; one of the world's most threatened sharks.",
        conservation_status="Critically Endangered",
        conservation_source=_IUCN,
        interesting_fact="It is one of the only sharks that lives permanently in fresh water and is known from remarkably few specimens.",
        detection="broad",
        detection_note="MegaFauna has a broad 'shark' class that cannot distinguish this species; no model is trained on Glyphis.",
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
        distribution="Arabian Sea and western Indian Ocean, especially the Malabar coast.",
        appearance="Small silvery-green fish, usually 10–20 cm, with a single dark spot behind the gill cover.",
        diet="Filter-feeder on plankton.",
        ecological_role="Cornerstone of the Indian coastal fishery and a vital link in the marine food web.",
        conservation_status="Least Concern",
        conservation_source=_IUCN,
        interesting_fact="Sardine shoals support huge traditional fisheries along the Kerala coast, with runs tied to the monsoon.",
        detection="broad",
        detection_note="Only a generic 'fish' class exists; none of the models is trained on Indian marine fish.",
    ),
    _sp(
        id="indian_mackerel",
        common_name="Indian mackerel",
        scientific_name="Rastrelliger kanagurta",
        group="Coastal & marine fish",
        habitat="Coastal / marine",
        distribution="Tropical Indo-Pacific, abundant along both Indian coasts.",
        appearance="Streamlined blue-green fish with dark wavy stripes, a forked tail and fine gill rakers; about 20–35 cm.",
        diet="Filter-feeder on plankton and small crustacean larvae.",
        ecological_role="Plankton-feeding forage fish; a major commercial catch in India.",
        conservation_status="Least Concern",
        conservation_source=_IUCN,
        interesting_fact="It feeds by swimming open-mouthed to sieve plankton from the water.",
        detection="broad",
        detection_note="Only a generic 'fish' class exists.",
    ),
    _sp(
        id="milkfish",
        common_name="Milkfish",
        scientific_name="Chanos chanos",
        group="Coastal & marine fish",
        habitat="Coastal / marine",
        distribution="Indo-Pacific; occurs in Indian coastal waters and is widely farmed.",
        appearance="Silvery, lozenge-shaped fish with a deeply forked tail and no teeth as an adult; grows to about 1.5 m.",
        diet="Omnivorous filter-feeder on algae and small invertebrates.",
        ecological_role="Hardy coastal fish and a mainstay of brackish-water aquaculture.",
        conservation_status="Least Concern",
        conservation_source=_IUCN,
        interesting_fact="It is the only living species of its family, Chanidae, a lineage with fossil relatives over 100 million years old.",
        detection="broad",
        detection_note="Generic 'fish' class only; the brackish model is not trained on Indian species.",
    ),
    _sp(
        id="spotted_seahorse",
        common_name="Spotted seahorse (yellow seahorse)",
        scientific_name="Hippocampus kuda",
        group="Coastal & marine fish",
        habitat="Coastal / marine",
        distribution="Indo-Pacific, including Indian coastal waters, seagrass beds and estuaries.",
        appearance="Small, heavily armoured fish with a horse-like head, curled prehensile tail and mottled brown-yellow colouring.",
        diet="Tiny crustaceans sucked up through a tubular snout.",
        ecological_role="Seagrass/mangrove dweller; traded for traditional medicine and aquaria, which drives overharvest.",
        conservation_status="Vulnerable",
        conservation_source=_IUCN,
        interesting_fact="Male seahorses become pregnant and give birth to the young.",
        detection="none",
        detection_note="No active model has a seahorse class.",
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
        distribution="Warm surface waters worldwide; regular along the Gujarat coast (Gulf of Kutch) and the Andaman Sea.",
        appearance="The largest fish in the world (up to ~18 m), a filter-feeder with a broad head and a white-spotted, checkerboard body.",
        diet="Filter-feeder on plankton and small schooling fish.",
        ecological_role="Gentle ocean giant; a major draw for ecotourism and a focus of Indian protection efforts.",
        conservation_status="Endangered",
        conservation_source=_IUCN,
        interesting_fact="Each whale shark's spot pattern is unique, like a fingerprint, and is used by researchers to identify individuals.",
        detection="broad",
        detection_note="MegaFauna and the aquarium model both have a broad 'shark' class, but neither distinguishes the whale shark from other sharks.",
    ),
    _sp(
        id="scalloped_hammerhead",
        common_name="Scalloped hammerhead",
        scientific_name="Sphyrna lewini",
        group="Shark & ray",
        habitat="Coastal / marine",
        distribution="Warm coastal and pelagic waters worldwide, including the Arabian Sea and Bay of Bengal.",
        appearance="Grey shark with a distinctive curved 'hammer' head bearing eyes at the tips; 2–3 m.",
        diet="Fish, squid, octopus and rays.",
        ecological_role="Coastal predator; heavily fished for its fins, which are highly valued.",
        conservation_status="Critically Endangered",
        conservation_source=_IUCN,
        interesting_fact="The wide head may improve the shark's electro-sensory detection of hidden prey.",
        detection="broad",
        detection_note="Only a broad 'shark' class exists; species identity is not established.",
    ),
    _sp(
        id="reef_manta_ray",
        common_name="Reef manta ray",
        scientific_name="Mobula alfredi",
        group="Shark & ray",
        habitat="Coastal / marine",
        distribution="Tropical Indo-Pacific reefs, including the Andaman and Lakshadweep seas.",
        appearance="Large diamond-shaped ray, up to ~5.5 m across, with horns beside the mouth and a white-and-dark dorsal pattern.",
        diet="Filter-feeder on zooplankton.",
        ecological_role="Keystone reef planktivore; a focal species for marine tourism.",
        conservation_status="Vulnerable",
        conservation_source=_IUCN,
        interesting_fact="Mantas are highly intelligent, appear to recognise themselves in mirrors and gather at cleaning stations.",
        detection="broad",
        detection_note="MegaFauna has a broad 'ray' class and the aquarium model a 'stingray' class, but neither identifies the species.",
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
        distribution="Tropical oceans worldwide; India hosts major mass-nesting beaches, notably Gahirmatha and Rushikulya in Odisha.",
        appearance="Small, olive-grey sea turtle with a heart-shaped shell; 60–70 cm.",
        diet="Omnivorous: jellyfish, crabs, shrimp and fish.",
        ecological_role="Long-distance migrant that links ocean and beach ecosystems; famous for synchronous mass nesting.",
        conservation_status="Vulnerable",
        conservation_source=_IUCN,
        interesting_fact="Odisha's beaches host arribadas — synchronised mass nestings where thousands of females come ashore together.",
        detection="broad",
        detection_note="MegaFauna has a broad 'turtle' class that cannot separate the Olive Ridley from other sea turtles.",
    ),
    _sp(
        id="green_sea_turtle",
        common_name="Green sea turtle",
        scientific_name="Chelonia mydas",
        group="Sea turtle",
        habitat="Coastal / marine",
        distribution="Tropical and subtropical seas, including the Lakshadweep and Andaman & Nicobar islands of India.",
        appearance="Large sea turtle with a smooth greenish-brown carapace and a single pair of scales between the eyes.",
        diet="Herbivorous as an adult, grazing seagrass and algae; juveniles are omnivorous.",
        ecological_role="Seagrass grazer that keeps meadows healthy.",
        conservation_status="Endangered",
        conservation_source=_IUCN,
        interesting_fact="Its green body fat, not its shell colour, gives the species its name.",
        detection="broad",
        detection_note="Only the broad MegaFauna 'turtle' class applies.",
    ),
    _sp(
        id="hawksbill_turtle",
        common_name="Hawksbill sea turtle",
        scientific_name="Eretmochelys imbricata",
        group="Sea turtle",
        habitat="Coastal / marine",
        distribution="Tropical reefs worldwide; nests on Indian islands and some mainland beaches.",
        appearance="Sea turtle with a beak-like mouth and overlapping, mottled amber-and-brown shell plates.",
        diet="Spongivore; feeds mainly on sea sponges on coral reefs.",
        ecological_role="Helps maintain reef health by controlling sponge populations.",
        conservation_status="Critically Endangered",
        conservation_source=_IUCN,
        interesting_fact="Its 'tortoiseshell' shell made it one of the most exploited turtles in history.",
        detection="broad",
        detection_note="Only the broad MegaFauna 'turtle' class applies.",
    ),
    _sp(
        id="leatherback_turtle",
        common_name="Leatherback sea turtle",
        scientific_name="Dermochelys coriacea",
        group="Sea turtle",
        habitat="Coastal / marine",
        distribution="All oceans; nests on some Indian beaches (for example in the Andaman & Nicobar Islands).",
        appearance="The largest sea turtle (up to ~2 m), with a leathery, ridged shell instead of hard scutes.",
        diet="Jellyfish specialist; also eats other soft-bodied prey.",
        ecological_role="Controls jellyfish populations across ocean basins.",
        conservation_status="Vulnerable",
        conservation_source=_IUCN,
        interesting_fact="It can dive deeper than 1,000 m and regulate its body temperature in cold water.",
        detection="broad",
        detection_note="Only the broad MegaFauna 'turtle' class applies; species identity is not established.",
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
        distribution="Indo-Pacific coral reefs, including the Lakshadweep and Andaman reefs of India.",
        appearance="Very large, thick-lipped reef fish reaching over 2 m, with a bulbous forehead that grows more pronounced with age; males are electric blue-green, females mottled.",
        diet="Carnivorous; crushes hard-shelled prey such as molluscs, sea stars and crustaceans.",
        ecological_role="Apex reef predator; one of the largest reef fishes, and naturally rare.",
        conservation_status="Endangered",
        conservation_source=_IUCN,
        interesting_fact="It changes sex during life — most large, colourful individuals are males that were born female.",
        detection="direct",
        detection_note="The Fish & Invertebrates model includes an explicit 'cheilinus_undulatus' class for this species.",
    ),
    _sp(
        id="humphead_parrotfish",
        common_name="Humphead parrotfish",
        scientific_name="Bolbometopon muricatum",
        group="Reef fish",
        habitat="Reef",
        distribution="Indo-Pacific reef slopes, including Indian reef areas.",
        appearance="The largest parrotfish (up to ~1.3 m) with a distinctive bulbous forehead and fused beak-like teeth.",
        diet="Herbivorous; scrapes live coral and algae, a major bioeroder of reefs.",
        ecological_role="Produces coral sand and shapes reef structure; ecologically vital and heavily overfished.",
        conservation_status="Vulnerable",
        conservation_source=_IUCN,
        interesting_fact="A single large individual can produce hundreds of kilograms of coral sand a year.",
        detection="direct",
        detection_note="The Fish & Invertebrates model includes an explicit 'bolbometopon_muricatum' class.",
    ),
    _sp(
        id="camouflage_grouper",
        common_name="Camouflage grouper (humpback grouper)",
        scientific_name="Cromileptes altivelis",
        group="Reef fish",
        habitat="Reef",
        distribution="Indo-Pacific reefs, including the Andaman and Nicobar islands.",
        appearance="Pale grey-white grouper with a high-arched back and a scattering of round black spots.",
        diet="Carnivorous; ambushes small fish and crustaceans among coral.",
        ecological_role="Reef predator; popular in the live food-fish trade, which pressures wild stocks.",
        conservation_status="Vulnerable",
        conservation_source=_IUCN,
        interesting_fact="Its spot pattern gives it the alternative name 'panther grouper'.",
        detection="direct",
        detection_note="The Fish & Invertebrates model includes an explicit 'cromileptes_altivelis' class.",
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
        distribution="Indo-Pacific shallow coastal flats, including Indian shores and the Gulf of Mannar.",
        appearance="Elongated, leathery sea cucumber, usually 15–30 cm, grey to olive with darker wrinkles.",
        diet="Deposit-feeder; swallows sediment and extracts organic matter.",
        ecological_role="Recycles nutrients and aerates seabed sediments; heavily harvested for the beche-de-mer trade.",
        conservation_status="Endangered",
        conservation_source=_IUCN,
        interesting_fact="When threatened it can expel its internal organs and later regrow them.",
        detection="broad",
        detection_note="The Fish & Invertebrates model has a generic 'sea_cucumber' class but does not identify the species.",
    ),
    _sp(
        id="mud_crab",
        common_name="Giant mud crab",
        scientific_name="Scylla serrata",
        group="Invertebrate",
        habitat="Brackish / estuarine",
        distribution="Indo-Pacific mangroves and estuaries, including India's Sundarbans and coastal backwaters.",
        appearance="Large, heavy crab with a smooth greenish-brown to nearly black shell and powerful claws.",
        diet="Omnivorous scavenger and predator of molluscs and small crabs.",
        ecological_role="Important mangrove/estuarine predator and scavenger; a valuable aquaculture species.",
        conservation_status=_NE,
        conservation_source=_NE,
        interesting_fact="Males grow much larger than females and are the main target of mud-crab aquaculture.",
        detection="broad",
        detection_note="The brackish model has a generic 'crab' class but cannot identify the species.",
    ),
    _sp(
        id="crown_of_thorns",
        common_name="Crown-of-thorns starfish",
        scientific_name="Acanthaster planci",
        group="Invertebrate",
        habitat="Reef",
        distribution="Indo-Pacific coral reefs, including Indian reefs.",
        appearance="Large multi-armed starfish up to ~35 cm covered in venomous spines.",
        diet="Coral; feeds by everting its stomach over coral polyps.",
        ecological_role="Natural coral predator; population outbreaks can devastate reefs.",
        conservation_status=_NE,
        conservation_source=_NE,
        interesting_fact="During outbreaks it can strip a reef of living coral, driving major reef declines.",
        detection="direct",
        detection_note="The Fish & Invertebrates model includes an explicit 'crown_of_thorns' class.",
    ),
)


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
