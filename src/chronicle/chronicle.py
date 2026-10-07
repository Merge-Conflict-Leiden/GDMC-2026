import base64
import logging
import os
import time

from dotenv import load_dotenv
from gdpc.interface import runCommand
from glm import ivec3
from openai import OpenAI, OpenAIError
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

load_dotenv()

logger = logging.getLogger(__name__)


def encode_image(image_path):
    with open(image_path, "rb") as image:
        return base64.b64encode(image.read()).decode("utf-8")


def escape(s: str) -> str:
    """Escape backslashes and double-quotes for the command string"""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def capture_seedmap_screenshot(
    seed: int,
    center_x: int,
    center_z: int,
    edition: str = "java",
    dimension: int = 0,
    output_path: str = "seedmap.png",
    wait_seconds: float = 6.0,
) -> str | None:
    """
    Open seedmap.app for the given seed and coordinates, wait for the map to
    render, take a screenshot, and return the file path.  Returns None on any
    error so callers can degrade gracefully.
    """
    url = (
        f"https://www.seedmap.app/seed/?seed={seed}"
        f"&edition={edition}"
        f"&dimension={dimension}"
        f"&centerX={center_x}"
        f"&centerZ={center_z}"
    )
    logger.info("Capturing seedmap screenshot from %s", url)

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1280,900")
    # Disable GPU in headless mode to avoid rendering glitches
    options.add_argument("--disable-gpu")

    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        driver.get(url)

        # Dismiss tutorial overlay if present.
        try:
            WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Skip')]"))
            ).click()
        except Exception:
            pass

        # Dismiss cookie consent banner if present.
        try:
            WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[contains(., 'Accept')]")
                )
            ).click()
        except Exception:
            pass

        try:
            marker = driver.find_element(By.ID, "coordMarkerToggle")
            if not marker.is_selected():
                marker.click()
        except Exception:
            logger.warning("Could not enable coordinate marker.")

        # Wait until at least one <canvas> element is present – seedmap.app
        # renders its map onto a canvas.
        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "canvas"))
            )
        except Exception:
            logger.warning(
                "capture_seedmap_screenshot: canvas element not found within timeout; "
                "screenshot may be incomplete."
            )

        # Give the map tiles additional time to finish rendering.
        time.sleep(wait_seconds)

        driver.save_screenshot(output_path)
        logger.info("Seedmap screenshot saved to %s", output_path)
        return output_path
    except Exception as exc:
        logger.error("capture_seedmap_screenshot failed: %s", exc)
        return None
    finally:
        if driver is not None:
            driver.quit()


class Chronicles:
    def __init__(
        self,
        editor,
        world_slice,
        location,
        width,
        length,
        center,
        villager_types,
        disaster,
        has_harbour,
        utility_buildings,
        lectern_pos,
        building_counts=None,
        castle_entrance=None,
        settlement_name=None,
        residents=None,
        districts=None,
        deceased=None,
        has_maze=False,
        has_construction=False,
        has_crypt=False,
        terrain_image_path="terrain.png",
        seedmap_image_path="seedmap.png",
    ):
        try:
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        except OpenAIError as e:
            logger.error("Failed to initialize OpenAI client: %s", e)
            self.client = None

        self.editor = editor
        self.heightmap = world_slice.heightmaps["MOTION_BLOCKING_NO_LEAVES"]
        self.width = width
        self.length = length

        seed: int | None = None
        try:
            _success, result_text = runCommand("seed")[0]
            if result_text:
                seed = int(result_text.split(": ")[1].strip("[]"))
        except Exception as exc:
            logger.warning("Chronicles: could not extract world seed (%s).", exc)
        self.seed = seed

        if center is None:
            self.center = ivec3(
                location[0] + width // 2,
                self.heightmap[width // 2, length // 2],
                location[1] + length // 2,
            )
            self.has_market = False
        else:
            self.center = ivec3(
                center[0], self.heightmap[center[0], center[1]], center[1]
            )
            self.has_market = True
        self.villager_types = villager_types
        self.disaster = disaster
        self.biome = world_slice.getPrimaryBiomeInChunk(self.center)
        self.nearest_biomes = []
        for x in range(0, width - 1, 16):
            for z in range(0, length - 1, 16):
                self.nearest_biomes[len(self.nearest_biomes) :] = (
                    world_slice.getBiomeCountsInChunk(
                        ivec3(x, self.heightmap[x, z], z)
                    ).keys()
                )
        self.nearest_biomes = list(set(self.nearest_biomes))
        # `has_harbour` is the actual placement result (True/False).  Accept a
        # legacy collection too (truthy iff non-empty) so older callers keep
        # working, but the caller should now pass whether a harbor was built —
        # a scheduled dock zone can be skipped as unbuildable.
        self.has_harbour = bool(has_harbour)
        self.utility_buildings = utility_buildings
        self.building_counts = building_counts or {}
        self.castle_entrance = castle_entrance
        self.settlement_name = settlement_name
        self.resident_labels = residents or []
        self.district_names = districts or []
        self.deceased_names = deceased or []
        self.has_maze = has_maze
        self.has_construction = has_construction
        self.has_crypt = has_crypt

        try:
            self.districts_image = encode_image(terrain_image_path)
        except FileNotFoundError:
            logger.warning(
                "Chronicles: terrain image %r not found; story will lack district map.",
                terrain_image_path,
            )
            self.districts_image = None

        self.seedmap_image: str | None = None
        if seed is not None:
            screenshot_path = capture_seedmap_screenshot(
                seed=seed,
                center_x=int(location[0]),
                center_z=int(location[1]),
                output_path=seedmap_image_path,
            )
            if screenshot_path is not None:
                try:
                    self.seedmap_image = encode_image(screenshot_path)
                except FileNotFoundError:
                    logger.warning(
                        "Chronicles: seedmap screenshot %r vanished before encoding.",
                        screenshot_path,
                    )

        self.lectern_pos = lectern_pos

    def _describe_settlement(self) -> str:
        """Build a structured description of the settlement."""

        sections = []

        if self.settlement_name:
            sections.append(f"Settlement name:\n{self.settlement_name}")

        sections.append(
            f"Location:\n"
            f"Primary biome: {self.biome.replace('_', ' ').title()}\n"
            f"Surrounding biomes: "
            f"{', '.join(b.replace('_', ' ').title() for b in self.nearest_biomes) if self.nearest_biomes else 'Unknown'}"
        )

        sections.append(
            f"Water access:\n"
            f"{'A working harbour provides access to trade by water.' if self.has_harbour else 'The settlement is inland with no harbour.'}"
        )

        if self.has_market:
            sections.append(
                "Marketplace:\n"
                "A central marketplace serves as the settlement's commercial heart."
            )

        if self.villager_types:
            sections.append(
                "Professions:\n"
                + ", ".join(v.replace("_", " ").title() for v in self.villager_types)
            )

        if self.resident_labels:
            sections.append(
                "Residents:\n" + "\n".join(f"- {name}" for name in self.resident_labels)
            )

        if self.district_names:
            sections.append(
                "Districts:\n"
                + "\n".join(f"- {district}" for district in self.district_names)
            )

        if self.utility_buildings:
            sections.append(
                "Important buildings:\n"
                + "\n".join(
                    f"- {building.replace('_', ' ').title()}"
                    for building in self.utility_buildings
                )
            )

        special = []

        if self.castle_entrance is not None:
            special.append("A fortified castle overlooks the settlement.")

        if self.has_maze:
            special.append("A hedge maze lies within the settlement.")

        if self.has_construction:
            special.append("Several buildings are currently under construction.")

        if self.has_crypt:
            special.append("A crypt beneath the graveyard holds honoured dead.")

        if special:
            sections.append(
                "Notable features:\n" + "\n".join(f"- {item}" for item in special)
            )

        if self.building_counts:
            sections.append(
                "Building composition:\n"
                + "\n".join(
                    f"- {count} {category.replace('_', ' ')}"
                    for category, count in sorted(self.building_counts.items())
                )
            )

        if self.disaster is not None:
            sections.append(
                f"Current disaster:\n{self.disaster.replace('_', ' ').title()}"
            )

        if self.deceased_names:
            sections.append(
                "Buried in the graveyard:\n"
                + "\n".join(f"- {name}" for name in self.deceased_names)
            )

        return "\n\n".join(sections)

    def _image_instructions(self) -> str:
        """Build the image-analysis part of the prompt, mentioning only the
        images that were actually captured and encoded."""

        descriptions = []

        if self.districts_image is not None:
            descriptions.append(
                "One image is an overhead render of the settlement: its districts, "
                "roads, and buildings, and the terrain beneath them — where the "
                "ground rises, where water lies, where trees stand thick."
            )

        if self.seedmap_image is not None:
            descriptions.append(
                "One image is a regional map centered on the settlement (marked "
                "with a coordinate marker), with pictograms for nearby villages, "
                "temples, ruins, and other landmarks. Read the wider landscape "
                "from it: coastlines, rivers, mountains, forests, and what lies "
                "in each direction. Use pictograms only if clearly visible."
            )

        if not descriptions:
            return ""

        plural = "s" if len(descriptions) > 1 else ""
        min_observations = 2 * len(descriptions)
        return (
            f"Inspect the attached image{plural} before writing.\n\n"
            + "\n\n".join(descriptions)
            + f"\n\nWeave in at least {min_observations} observations that could "
            f"only come from the image{plural}; do not invent features that are "
            "not visible.\n\n"
        )

    def prompt_model(self) -> str:
        """Build the prompt and call the model; return the generated story text."""

        settlement_description = self._describe_settlement()

        system_instruction = (
            "You are a skilled medieval author. Your document will be displayed on a "
            "lectern in the town center of the settlement described below, and should "
            "feel as though it genuinely belongs there.\n\n"
            "Write believable medieval prose, not high fantasy: ordinary people, local "
            "traditions, commerce, religion, memory, and daily life.\n\n"
            "Build only on the supplied facts. Do not invent major landmarks, "
            "geography, disasters, rulers, or inhabitants; leave unknown details "
            "unspecified.\n\n"
            "Never mention Minecraft, blocks, coordinates, chunks, biomes, or game "
            "mechanics. Render biome names as a local would describe the land (a "
            "'Snowy Taiga' is cold pine country where winters bite).\n\n"
            "Output only the finished document — no title, heading, author line, or "
            "explanatory text. Approximately 300-600 words."
        )

        prompt = (
            "Write a document that could naturally lie on the lectern in this "
            "settlement's town center: a chronicle, traveller's journal, letter, "
            "proclamation, merchant's record, diary, or similar.\n\n"
            "Center it on a single topic — an event, concern, celebration, journey, "
            "or memory — rather than describing everything. Assume the settlement "
            "has existed for generations; traditions, customs, and remembered "
            "events are welcome.\n\n"
            "Tie it unmistakably to this settlement: weave in at least six concrete "
            "facts from the description below, showing how they shape daily life "
            "rather than listing them. Use the supplied names of people, districts, "
            "and the deceased as if they matter to the writer.\n\n"
            "Ground the document in the terrain and surroundings: the land "
            "underfoot, what can be seen from the settlement, and how the "
            "landscape shapes work, food, trade, travel, and safety.\n\n"
            f"{self._image_instructions()}"
            f"SETTLEMENT\n"
            f"{settlement_description}"
        )

        content: list[dict] = [{"type": "input_text", "text": prompt}]

        # District / terrain image
        if self.districts_image is not None:
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{self.districts_image}",
                }
            )

        # Seedmap regional overview
        if self.seedmap_image is not None:
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{self.seedmap_image}",
                }
            )

        response = self.client.responses.create(
            model="gpt-5",
            instructions=system_instruction,
            input=[{"role": "user", "content": content}],
        )
        return response.output_text

    def _paginate(self, text: str, max_chars: int = 255) -> list[str]:
        """Split text into page-sized chunks on word boundaries."""
        words = text.split()
        pages, current = [], ""
        for word in words:
            if len(current) + len(word) + 1 > max_chars:
                pages.append(current.strip())
                current = word
            else:
                current += (" " if current else "") + word
        if current:
            pages.append(current.strip())
        return pages

    def build_lectern_command(self, x, y, z, pages) -> str:
        pages_nbt = ", ".join(f'"{escape(p)}"' for p in pages)
        return (
            f"data modify block {x} {y} {z} Book set value "
            f'{{id:"minecraft:written_book",count:1,'
            f'components:{{"minecraft:written_book_content":'
            f'{{title:"Chronicle",author:"Merge Conflict 2",'
            f"generation:0,pages:[{pages_nbt}]}}}}}}"
        )

    def place_chronicle(self):
        if self.lectern_pos is None:
            logger.warning("place_chronicle: no lectern position, skipping.")
            return
        if self.client is None:
            logger.warning("place_chronicle: no OpenAI client, skipping.")
            return

        try:
            text = self.prompt_model()
        except Exception:
            logger.exception("place_chronicle: prompt_model failed, skipping.")
            return

        pages = self._paginate(text)
        print(pages)
        cmd = self.build_lectern_command(
            self.lectern_pos[0], self.lectern_pos[1], self.lectern_pos[2], pages
        )
        self.editor.runCommand(cmd)
