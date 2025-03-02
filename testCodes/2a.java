import static org.junit.Assert.*;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

public class AlertAdProxyTest {
    private final ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
    private final PrintStream originalOut = System.out;
    private AlertAdProxy adProxy;

    @Before
    public void setUp() {
        System.setOut(new PrintStream(outputStream));
        adProxy = new AlertAdProxy();
    }

    @After
    public void tearDown() {
        System.setOut(originalOut);
    }

    @Test
    public void testShowBannerAd() {
        adProxy.showAd("Banner", "This is a banner ad.");
        assertTrue(outputStream.toString().contains("Showing Banner Ad: This is a banner ad."));
    }

    @Test
    public void testShowInterstitialAd() {
        adProxy.showAd("Interstitial", "This is an interstitial ad.");
        assertTrue(outputStream.toString().contains("Showing Interstitial Ad: This is an interstitial ad."));
    }

    @Test
    public void testShowVideoAd() {
        adProxy.showAd("Video", "This is a video ad.");
        assertTrue(outputStream.toString().contains("Showing Video Ad: This is a video ad."));
    }

    @Test
    public void testInvalidAdType() {
        adProxy.showAd("InvalidType", "This is an invalid ad.");
        assertTrue(outputStream.toString().contains("Invalid Ad Type!"));
    }
}
