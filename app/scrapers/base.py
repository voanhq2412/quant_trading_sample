import subprocess
import time
from typing import Any, List

from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

from app.utils import write_to_log


class ScrapeToolbox:
    """
    Here you find methods that can are common and usable across multipler scrapers
    such as starting driver, closing driver, and some processing methods
    """

    LOG_NAME = "scraper_toolbox"
    DRIVER = None

    # Function to (re)start driver
    def start_driver(self, proxy, force_restart=False) -> None:
        if force_restart:
            self.close_driver()
        # Setting up the driver
        options = webdriver.ChromeOptions()
        ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
        options.add_argument("-headless=new")
        options.add_argument("-no-sandbox")
        options.add_argument("-disable-dev-shm-usage")
        options.add_argument(f"user-agent={ua}")
        if proxy == "tor":
            options.add_argument("--proxy-server=socks5://localhost:9050")
        elif proxy == "warp":
            options.add_argument("--proxy-server=socks5://localhost:9060")
        self.DRIVER: webdriver.chrome.webdriver.WebDriver = webdriver.Chrome(
            service=Service(executable_path=r"config/chromedriver"), options=options
        )
        self.DRIVER.implicitly_wait(3)

    # Wrapper to close driver if its created
    def close_driver(self) -> None:
        if self.DRIVER is not None:
            self.DRIVER.quit()

    """
        Versatile method:
        Usage
        1) to fetch link: provide link
        2) To extract element via xpath: provide web elem + xpath
        Parameters
        t1 = WAIT TIME FOR ELEMENT/LINK
        t2 = WAIT TIME AFTER EACH ERROR
        max_attempt = number of times to retry
    """

    def attack(self, target, xpath=None, t1=1, t2=5, max_attempt=2) -> List[Any]:
        attempt = 1
        while attempt <= max_attempt:
            try:
                if type(target) == str:
                    write_to_log(self.LOG_NAME, f"Scraping from {target}")
                    self.DRIVER.get(target)
                    time.sleep(t1)
                    break
                elif (
                    type(target) == webdriver.chrome.webdriver.WebDriver
                    or type(target) == webdriver.remote.webelement.WebElement
                ):
                    assert xpath is not None
                    WebDriverWait(self.DRIVER, t1).until(
                        EC.presence_of_element_located((By.XPATH, xpath))
                    )
                    elem = target.find_elements("xpath", xpath)
                    return elem
            except TimeoutException:
                if attempt == max_attempt:
                    write_to_log(self.LOG_NAME, "Timeout error - Cannot Resolve")
                    break
                else:
                    time.sleep(t2 * attempt)
                    attempt += 1
                    continue
            except StaleElementReferenceException:
                if attempt == max_attempt:
                    write_to_log(
                        self.LOG_NAME,
                        "StaleElementReference error - Cannot Resolve",
                    )
                    break
                else:
                    time.sleep(t2 * attempt)
                    attempt += 1
                    continue
            except WebDriverException:
                if attempt == max_attempt:
                    write_to_log(
                        self.LOG_NAME,
                        "WebDriver error - Cannot Resolve",
                    )
                    break
                else:
                    time.sleep(t2 * attempt)
                    attempt += 1
                    continue
            except NoSuchElementException:
                if attempt == max_attempt:
                    write_to_log(
                        self.LOG_NAME,
                        "NoSuchElement error - Cannot Resolve",
                    )
                    break
                else:
                    time.sleep(t2 * attempt)
                    attempt += 1
                    continue
            except Exception as ex:
                message = f"""Unidentified error:
                {type(ex)} : {ex.args} """
                write_to_log(self.LOG_NAME, message)
                break
        return []

    """
        TOR SERVICE
    """

    def start_tor(self):
        cmd = ["sudo", "service", "tor", "start"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE)
        print(result.stdout.decode("utf-8"))

    def stop_tor(self):
        cmd = ["sudo", "service", "tor", "stop"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE)
        print(result.stdout.decode("utf-8"))

    """
        WARP SERVICE
    """

    def start_warp(self):
        cmds = [
            ["sudo", "service", "warp-svc", "start"],
            ["warp-cli", "set-mode", "proxy"],
            ["warp-cli", "set-proxy-port", "9060"],
            ["warp-cli", "connect"],
        ]
        for cmd in cmds:
            result = subprocess.run(cmd, stdout=subprocess.PIPE)
            time.sleep(1)
            print(result.stdout.decode("utf-8"))

    def stop_warp(self):
        cmds = [
            ["warp-cli", "disconnect"],
            ["sudo", "service", "warp-svc", "stop"],
        ]
        for cmd in cmds:
            result = subprocess.run(cmd, stdout=subprocess.PIPE)
            print(result.stdout.decode("utf-8"))
