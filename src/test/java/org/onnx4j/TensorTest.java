/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.onnx4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

import java.lang.ref.PhantomReference;
import java.lang.ref.ReferenceQueue;
import java.nio.ByteBuffer;

import org.junit.Test;
import org.onnx4j.tensor.DataType;
import org.onnx4j.tensor.Shape;
import org.onnx4j.tensor.TensorBuilder;
import org.onnx4j.utils.BufferUtil;
import org.onnx4j.utils.UnsafeAccess;

/**
 * Unit test for Tensor.class
 */
public class TensorTest {

	/**
	 * @throws Exception
	 * 
	 */
	@Test
	public void testToString() throws Exception {
		TensorManager<Tensor> tsMgr = new TensorManager<Tensor>() {

			@Override
			protected void dispose(Tensor tensor) {
				tensor.close();
			}

		};
		TensorBuilder builder = TensorBuilder.builder(DataType.FLOAT, Shape.create(2L, 3L, 3L), Tensor.options())
				.manager(tsMgr);
		for (int n = 0; n < 2 * 3 * 3; n++) {
			builder.putFloat(new Float(n));
		}

		try (Tensor ts = builder.build()) {
			String expectedOutput = "Tensor[2, 3, 3] = [[[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0],],[[9.0,10.0,11.0],[12.0,13.0,14.0],[15.0,16.0,17.0],],]";
			String actualOutput = ts.toString().replaceAll("[\t\n]", "");
			assertEquals(expectedOutput, actualOutput);
			System.out.println(actualOutput);
		}
	}

	@SuppressWarnings("restriction")
	@Test
	public void testDirectMemAllocationAndDeallocation() {
		ByteBuffer byteBuffer = ByteBuffer.allocateDirect(1024 * 1024 * 1024);
		System.out.println("Allocated " + byteBuffer.capacity() + " bytes memory");
		// DirectBufferDealloc.deallocateDirectBuffer(byteBuffer);
		UnsafeAccess.UNSAFE.freeMemory(BufferUtil.address(byteBuffer));
		System.out.println("Allocated " + byteBuffer.capacity() + " bytes memory");
	}

	@Test
	public void testWrapedByPhantomReference() {
		TensorManager<Tensor> tsMgr = new TensorManager<Tensor>() {

			@Override
			protected void dispose(Tensor tensor) {
				tensor.close();
			}

		};
		TensorBuilder builder = TensorBuilder.builder(DataType.FLOAT, Shape.create(2L, 3L, 3L), Tensor.options())
				.manager(tsMgr);
		Tensor tensor = builder.build();
		assertEquals(
				"Tensor[2, 3, 3] = [[[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],],[[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],],]",
				tensor.toString().replaceAll("[\t\n]", ""));

		ReferenceQueue<Tensor> queue = new ReferenceQueue<Tensor>();
		PhantomReference<Tensor> pf = new PhantomReference<Tensor>(tensor, queue);
		assertNull(pf.get());
		assertNull(queue.poll());

		// tensor = null;
		while (true) {
			System.gc();
			// assertNotNull(queue.poll());
			if (queue.poll() != null)
				break;
			else {
				/*
				 * try { Thread.sleep(500); } catch (InterruptedException e) {
				 * // TODO Auto-generated catch block e.printStackTrace(); }
				 */
			}
		}

		// tensor = null;
		// System.gc();
		// assertNull(tensor);
		// System.out.println(tensor.toString());
	}

}