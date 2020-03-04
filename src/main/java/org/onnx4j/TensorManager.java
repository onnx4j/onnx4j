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

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class TensorManager<T_TS> implements AutoCloseable {

	private static Logger logger = LoggerFactory.getLogger(TensorManager.class);

	protected abstract void dispose(T_TS tensor);

	private boolean hasClosed = false;
	private Map<String, T_TS> tensors = new HashMap<String, T_TS>();

	public void attach(String name, T_TS tensor) {
		this.tensors.put(name, tensor);
	}

	public void detach(String name) {
		this.tensors.remove(name);
	}

	public T_TS get(String name) {
		return this.tensors.get(name);
	}

	public Map<String, T_TS> get() {
		return Collections.unmodifiableMap(this.tensors);
	}

	@Override
	public void close() throws Exception {
		if (this.hasClosed)
			throw new IllegalStateException("The TensorManager has closed.");

		for (Entry<String, T_TS> entry : this.tensors.entrySet()) {
			try {
				this.dispose(entry.getValue());
				logger.debug("Tensor[{}:{}] has been released.", entry.getValue().getClass().getName(),
						entry.getKey());
			} catch (Exception e) {
				logger.error("Tensor[{}:{}] can not be released.", entry.getValue().getClass().getName(),
						entry.getKey());
			}
			
		}

		this.tensors.clear();
		this.tensors = null;

		this.hasClosed = true;
	}

}
